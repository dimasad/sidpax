"""Smoother-Error Method."""

import collections
from dataclasses import dataclass, InitVar
import operator
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax_dataclasses as jdc
from jax.flatten_util import ravel_pytree

from . import mat, sparse, stats
from .sparse import sparse_objective


@jdc.pytree_dataclass
class Data:
    y: jax.Array
    """Measured outputs."""

    u: jax.Array
    """Exogenous inputs."""

    def __len__(self):
        """Number of samples."""
        assert len(self.y) == len(self.u)
        return len(self.y)


@dataclass
class Estimator:

    model: Any
    """Underlying dynamical system model."""

    @jdc.pytree_dataclass
    class Param:
        p: Any
        mu: jax.Array
        Sigma_cond: mat.PositiveDefiniteMatrix
        S_cross: jax.Array

    def param(self, data, rng=None):
        # Get base sizes
        nx = self.model.nx
        ny = self.model.ny
        N = len(data)

        # Initialize parameters
        p = self.model.param(data, rng)
        mu = jnp.zeros((N, nx))
        Sigma_cond = mat.LExpDLT.identity(nx)
        S_cross = jnp.zeros((self.model.nx, nx))

        # Create dataclass and return
        return self.Param(p=p, mu=mu, Sigma_cond=Sigma_cond, S_cross=S_cross)

    @sparse_objective
    @staticmethod
    def state_path_entropy(Sigma_cond: mat.PositiveDefiniteMatrix, N: int):
        """Differential entropy of the state-path posterior."""
        return 0.5 * Sigma_cond.logdet * N

    @state_path_entropy.set_param_filter
    def state_path_entropy(self, param: Param) -> jax.Array:
        """Selects parameters of the state-path posterior entropy."""
        return param.Sigma_cond

    @jdc.pytree_dataclass
    class PriorParam:
        """Prior parameters."""

        p: Any
        mu0: jax.Array
        Sigma_cond: mat.PositiveDefiniteMatrix
        S_cross: jax.Array

    @sparse_objective
    def prior_logpdf(self, param: PriorParam):
        """Log-density of the initial state and parameters."""
        x_samples = self.sample_x_marg(
            param.Sigma_cond, param.S_cross, param.mu0
        )
        return self.model.bind(param.p).prior_logpdf(x_samples).sum()

    @prior_logpdf.set_param_filter
    def prior_logpdf(self, param: Param) -> PriorParam:
        """Selects parameters of the prior."""
        return self.PriorParam(
            p=param.p,
            mu0=param.mu[0],
            Sigma_cond=param.Sigma_cond,
            S_cross=param.S_cross,
        )

    @jdc.pytree_dataclass
    class TransParam:
        """State transition parameters."""

        mu_curr: jax.Array
        mu_next: jax.Array
        p: Any
        Sigma_cond: mat.PositiveDefiniteMatrix
        S_cross: jax.Array

    @sparse_objective
    def trans_logpdf(self, param: TransParam, u_curr: jax.Array):
        # Get the Cholesky factor of the marginal state covariance
        S_cond = param.Sigma_cond.chol_low
        S_marg = mat.tria_qr(S_cond, param.S_cross)

        # Get the standard normal sigma points and weights.
        # Note that weights will be discarded, as they are all equal.
        nx = self.model.nx
        std_dev, weights = stats.sigmapts(2 * nx)

        # Scale the sigma points
        x_curr_dev = jnp.matvec(S_marg, std_dev[:, :nx])
        x_next_dev = jnp.matvec(param.S_cross, std_dev[:, :nx]) + jnp.matvec(
            S_cond, std_dev[:, -nx:]
        )

        # Add the mean
        x_curr_samples = param.mu_curr + x_curr_dev
        x_next_samples = param.mu_next + x_next_dev

        # Bind the model to the parameters and compute the logpdf
        mdl = self.model.bind(param.p)
        return mdl.trans_logpdf(x_next_samples, x_curr_samples, u_curr).mean()

    @trans_logpdf.set_param_filter
    def trans_logpdf(self, param: Param) -> TransParam:
        return self.TransParam(
            mu_curr=param.mu[:-1],
            mu_next=param.mu[1:],
            p=param.p,
            Sigma_cond=param.Sigma_cond,
            S_cross=param.S_cross,
        )

    trans_logpdf.vmap_in_axes = (TransParam(0, 0, None, None, None), 0)

    @sparse_objective
    def meas_logpdf(self, param: Param, data: Data):
        x_samples = self.sample_x_marg(
            param.Sigma_cond, param.S_cross, param.mu
        )

        # Bind the model to the parameters and compute the logpdf
        mdl = self.model.bind(param.p)
        return mdl.meas_logpdf(data.y, x_samples, data.u).mean()

    meas_logpdf.vmap_in_axes = (Param(None, 0, None, None), 0)

    def elbo(self, param: Param, data: Data) -> jax.Array:
        """Variational inference evidence lower bound."""
        prior = self.prior_logpdf(param)
        entropy = self.state_path_entropy(param, len(data))
        trans_logpdf = self.trans_logpdf(param, data.u[:-1])
        meas_logpdf = self.meas_logpdf(param, data)
        return prior + entropy + trans_logpdf.sum(0) + meas_logpdf.sum(0)

    def elbo_hessian(self, param: Param, data: Data, param_ind=None):
        """Hessian of the ELBO with respect to the parameters."""
        kwd = dict(param_ind=param_ind)
        hess_components = [
            self.prior_logpdf.hessian(param, **kwd),
            self.state_path_entropy.hessian(param, len(data), **kwd),
            self.trans_logpdf.hessian(param, data.u[:-1], **kwd),
            self.meas_logpdf.hessian(param, data, **kwd),
        ]
        return sparse.concatenate_coo(*hess_components)

    def sample_x_marg(self, Sigma_cond, S_cross, mu):
        """Sample from the marginal state distribution."""
        # Get the Cholesky factor of the marginal state covariance
        S_marg = mat.tria_qr(Sigma_cond.chol_low, S_cross)

        # Get the standard normal sigma points and weights.
        # Note that weights will be discarded, as they are all equal.
        nx = self.model.nx
        std_dev, weights = stats.sigmapts(nx)

        # Scale the sigma points
        x_dev = jnp.matvec(S_marg, std_dev)

        # Add the mean and return
        x_samples = mu + x_dev
        return x_samples


@dataclass
class MultiSegmentProblem:
    data: list[Data]
    estimators: list[Estimator]
    is_unique: Estimator.Param = None

    def __post_init__(self):
        if isinstance(self.estimators, Estimator):
            self.estimators = len(self.data) * [self.estimators]
        if self.is_unique is None:
            self.is_unique = Estimator.Param(
                p=True, mu=False, S_cross=False, Sigma_cond=False
            )

    def param(self, rng=None):
        unmerged = [
            e.params(d, rng) for e, d in zip(self.estimators, self.data)
        ]
        return merge_trees(self.is_unique, *unmerged)
    


@jdc.pytree_dataclass
class MergedPyTree:
    """Sequence of pytrees split into unique and replicated leaves and subtrees.

    The unique subtrees, leaves, and leaf elements are the same for all sequence
    elements. Global is also a good name for them. The replicated subtrees,
    leaves, and leaf elements are replicated for each sequence element (local).
    """

    unique: Any
    """pytree with unique elements or `None` if they are replicated."""

    replicated: list
    """List of pytrees with replicated elements or `None` if they are unique."""

    is_unique: jdc.Static[Any]
    """pytree with `bool` or `jax.Array` of `bool` specifying what is unique."""

    def __getitem__(self, index: int):
        return jax.tree.map(
            leaf_where, self.is_unique, self.unique, self.replicated[index]
        )

    def __len__(self):
        """Number of merged trees."""
        return len(self.replicated)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def leaf_select(condition, param):
    if isinstance(condition, jax.Array):
        return param[condition.astype(bool)]
    elif condition:
        return param


def leaf_select_not(condition, param):
    if isinstance(condition, jax.Array):
        return param[~condition.astype(bool)]
    elif not condition:
        return param


def leaf_asfloat(condition):
    if isinstance(condition, jax.Array):
        return condition.astype(float)
    else:
        return float(condition)


def leaf_where(condition, x, y):
    if isinstance(condition, jax.Array):
        return jnp.where(condition, x, y)
    else:
        return x if condition else y


def pytree_asfloat(condition):
    return jax.tree.map(leaf_asfloat, condition)


def merge_trees(is_unique, *trees) -> MergedPyTree:
    """
    Merges a sequence of pytrees into unique and replicated leaves and subtrees.

    Parameters
    ----------
    is_unique : pytree with `bool` or `jax.Array` of `bool` as leaves.

    *trees : sequence of one or more pytrees
        The trees must have same structure as `is_unique` or have it as a
        prefix.

    Returns
    -------
    merged : MergedPyTree
        Dataclass reprenting the merging, with the unique taken from `trees[0]`.

    Examples
    --------
    >>> from sidpax.sem import merge_trees
    >>> import jax.numpy as jnp
    >>> trees = [
    ...     dict(a=1.0, b=jnp.array([2.0, 3.0]), c=[4.0, 5.0]),
    ...     dict(a=6.0, b=jnp.array([7.0, 8.0]), c=[9.0, 10.0]),
    ... ]
    >>> is_unique = dict(a=True, b=jnp.array([True, False]), c=False)
    >>> merged = merge_trees(is_unique, *trees)

    >>> # unique has only the leaves or subtrees specified in `is_unique`
    >>> merged.unique
    {'a': 1.0, 'b': Array([2.], dtype=...), 'c': None}

    >>> # replicated has each sequence element's non-unique leaves or subtrees
    >>> merged.replicated[0]
    {'a': None, 'b': Array([3.], dtype=...), 'c': [4.0, 5.0]}
    >>> merged.replicated[1]
    {'a': None, 'b': Array([8.], dtype=...), 'c': [9.0, 10.0]}

    >>> # The merged sequence elements can be accessed with indexing
    >>> merged[0]
    {'a': 1.0, 'b': Array([2., 3.], dtype=...), 'c': [4.0, 5.0]}
    """
    if len(trees) == 0:
        raise TypeError("At least one tree required.")

    is_unique_f = pytree_asfloat(is_unique)
    unique = jax.tree.map(leaf_select, is_unique_f, trees[0])
    replicated = [jax.tree.map(leaf_select_not, is_unique_f, t) for t in trees]
    return MergedPyTree(
        unique=unique, replicated=replicated, is_unique=is_unique
    )
