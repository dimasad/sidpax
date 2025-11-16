"""Smoother-Error Method."""

import copy
import dataclasses
import functools
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import hedeut
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax_dataclasses as jdc
from jax.flatten_util import ravel_pytree

from . import common, mat, stats


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
class SparseObjective:
    """A sparse component of the objective function."""

    fun: Callable
    """The underlying objective function."""

    param_filter: Callable = lambda obj, x: x
    """Filters which parameters are used in this component of the objective."""

    vmap_in_axes: None | int | Sequence[Any] = None
    """Axes specification for vectorization over the inputs of bound `fun`."""

    vmap_out_axes: None | int | Sequence[Any] = 0
    """Axes specification for vectorization over the outputs of `fun`."""

    obj: None | object = None
    """The object the objective function is bound to (None if unbound)."""

    @property
    def _bound_fun(self):
        """`fun` bound to the underlying object."""
        return self.fun.__get__(self.obj, type(self.obj))

    @property
    def _bound_param_filter(self):
        """`param_filter` bound to the underlying object."""
        return self.param_filter.__get__(self.obj, type(self.obj))

    def __call__(self, param, *args):
        """Binds `fun` to `obj`, vmaps it, filters the first arg, and calls."""
        if self.obj is None:
            raise RuntimeError("Cannot call unbound SparseObjective.")

        # Bind method to object
        fun = self._bound_fun

        # Vectorize if needed
        if self.vmap_in_axes is not None:
            fun = jax.vmap(fun, self.vmap_in_axes, self.vmap_out_axes)

        # Get the function parameters
        fun_param = self._bound_param_filter(param)

        # Call the bound and vectorized method
        return fun(fun_param, *args)

    def __get__(self, obj, objtype=None):
        """Returns a copy"""
        if obj is None:
            raise TypeError
        return dataclasses.replace(self, obj=obj)

    def set_param_filter(self, param_filter):
        """Sets the parameter filter function, for use as a decorator."""
        return dataclasses.replace(self, param_filter=param_filter)

    def hessian(self, param, *args, param_ind=None):
        """Return the Hessian of `fun` with respect to the filtered params."""
        if self.obj is None:
            raise RuntimeError("Hessian requires bound SparseObjective.")

        # Create parameters if needed
        if param_ind is None:
            param_ind = common.pytree_ind(param)

        # Get the function parameters and parameter indices
        fun_param = self._bound_param_filter(param)
        fun_param_ind = self._bound_param_filter(param_ind)

        # Obtain sparse Hessian function
        hess = common.sparse_hessian(
            self._bound_fun, 0, self.vmap_in_axes, self.vmap_out_axes
        )

        return hess((fun_param, *args), (fun_param_ind,))


def sparse_objective(fun: Callable) -> SparseObjective:
    """Decorator for creating a `SparseObjective` in a class, from a method.

    Parameters
    ----------
    fun
        The method implementing the objective function that will be wrapped.
        If `fun` is a static or class method, it should be wrapped with the
        appropriate decorator before being passed to this function.

    Examples
    --------

    >>> import jax
    >>> import numpy as np
    >>> import scipy.sparse
    >>> from sidpax.sem import sparse_objective

    >>> # Create a simple problem with a decorated objective function
    >>> class ExampleProblem:
    ...     @sparse_objective
    ...     def obj(self, param_filtered: jax.Array, y: jax.Array):
    ...         return 0.5 * (param_filtered ** 2).sum(0) * y
    ...
    ...     @obj.set_param_filter
    ...     def obj(self, param):
    ...         return param["filtered"]
    ...
    ...     # Vectorize only over first axis of filtered parameters
    ...     obj.vmap_in_axes = (0, None) 

    >>> # Create the parameters
    >>> param = dict(filtered=jax.numpy.asarray([[1, 2],[3, 4]], float), z=5.0)
    >>> y = jax.numpy.asarray(2.0)

    >>> # Instantiate the problem and compute the objective
    >>> problem = ExampleProblem()
    >>> obj_value = problem.obj(param, y)
    >>> print(*obj_value)
    5.0 25.0

    >>> # Compute the Hessian
    >>> hess_coo = problem.obj.hessian(param, y)
    >>> hess = scipy.sparse.coo_array(hess_coo).todense()
    >>> np.testing.assert_allclose(hess, np.identity(4) * y)
    """
    return functools.wraps(fun)(SparseObjective(fun))


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
        return common.concatenate_coo(*hess_components)

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
