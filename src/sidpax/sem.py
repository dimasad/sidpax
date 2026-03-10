"""Smoother-Error Method."""

from dataclasses import dataclass
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import scipy.sparse

from sidpax.tree import merge_trees

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
class SegmentProblem:

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
class Estimator:
    data: list[Data]
    subproblems: SegmentProblem | list[SegmentProblem]
    is_unique: SegmentProblem.Param = SegmentProblem.Param(
        p=True, mu=False, S_cross=False, Sigma_cond=False
    )
    fix_p: bool = False

    def param(self, rng=None):
        param_list = [
            p.param(d, rng) for p, d in zip(self._subproblems, self.data)
        ]
        if self.fix_p:
            param_list = [jdc.replace(param, p={}) for param in param_list]
        return merge_trees(self.is_unique, *param_list)

    @property
    def _subproblems(self) -> list[SegmentProblem]:
        """A list with the subproblems."""
        if isinstance(self.subproblems, Sequence):
            return self.subproblems
        else:
            return [self.subproblems] * len(self.data)

    def cost(self, paramvec):
        params = self.unpack(paramvec)
        cost = 0.0
        for prob, param, data in zip(self._subproblems, params, self.data):
            cost = cost - prob.elbo(param, data)
        return cost

    def grad(self, paramvec):
        return jax.grad(self.cost)(paramvec)

    @property
    def sparse_hessian_fun(self):
        hessian_coo = jax.jit(self.hessian_coo)
        return lambda v: scipy.sparse.coo_array(hessian_coo(v))

    def hessian_coo(self, paramvec):
        params = self.unpack(paramvec)
        inds = sparse.pytree_ind(params)
        coo_list = [
            prob.elbo_hessian(p, d, i)
            for prob, p, d, i in zip(self._subproblems, params, self.data, inds)
        ]
        elbo_coo = sparse.concatenate_coo(*coo_list)
        cost_coo = -elbo_coo[0], elbo_coo[1]
        return cost_coo

    def unpack(self, paramvec):
        rng = jax.random.key(0)
        unpack = jax.flatten_util.ravel_pytree(self.param(rng))[1]
        return unpack(paramvec)
