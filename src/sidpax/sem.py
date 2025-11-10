"""Smoother-Error Method."""

import copy
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
    """Axes specification for vectorization over inputs."""

    vmap_out_axes: None | int | Sequence[Any] = 0
    """Axes specification for vectorization over outputs."""

    obj: Any = None
    """The object the objective function is bound to."""

    @property
    def bound_fun(self):
        """`fun` bound to the underlying object"""
        return self.fun.__get__(self.obj, type(self.obj))

    @property
    def bound_param_filter(self):
        """`param_filter` bound to the underlying object"""
        return self.param_filter.__get__(self.obj, type(self.obj))

    def __call__(self, param, *args):
        if self.obj is None:
            raise RuntimeError("Cannot call unbound SparseObjective.")

        # Bind method to object
        fun = self.bound_fun

        # Vectorize if needed
        if self.vmap_in_axes is not None:
            fun = jax.vmap(fun, self.vmap_in_axes, self.vmap_out_axes)

        # Get the function parameters
        fun_param = self.bound_param_filter(param)

        # Call the bound and vectorized method
        return fun(fun_param, *args)

    def __get__(self, obj, objtype=None):
        if obj is None:
            raise TypeError

        self_copy = copy.copy(self)
        self_copy.obj = obj
        return self_copy

    def param_filter_fun(self, param_filter):
        self.param_filter = param_filter
        return self

    def hessian(self, param, *args, param_ind=None):
        if self.obj is None:
            raise RuntimeError("Hessian requires bound SparseObjective.")

        # Create parameters if needed
        if param_ind is None:
            param_ind = common.pytree_ind(param)

        # Get the function parameters and parameter indices
        fun_param = self.bound_param_filter(param)
        fun_param_ind = self.bound_param_filter(param_ind)

        # Obtain sparse Hessian function
        hess = common.sparse_hessian(
            self.bound_fun, 0, self.vmap_in_axes, self.vmap_out_axes
        )

        return hess((fun_param, *args), (fun_param_ind,))


def sparse_objective(fun):
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

    @state_path_entropy.param_filter_fun
    def state_path_entropy(self, param: Param) -> jax.Array:
        """Selects parameters of the state-path posterior entropy."""
        return param.Sigma_cond

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

    @trans_logpdf.param_filter_fun
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
        # Get the Cholesky factor of the marginal state covariance
        S_cond = param.Sigma_cond.chol_low
        S_marg = mat.tria_qr(S_cond, param.S_cross)

        # Get the standard normal sigma points and weights.
        # Note that weights will be discarded, as they are all equal.
        nx = self.model.nx
        std_dev, weights = stats.sigmapts(nx)

        # Scale the sigma points
        x_dev = jnp.matvec(S_marg, std_dev)

        # Add the mean
        x_samples = param.mu + x_dev

        # Bind the model to the parameters and compute the logpdf
        mdl = self.model.bind(param.p)
        return mdl.meas_logpdf(data.y, x_samples, data.u).mean()

    meas_logpdf.vmap_in_axes = (Param(None, 0, None, None), 0)

    def elbo(self, param: Param, data: Data) -> jax.Array:
        """Variational inference evidence lower bound."""
        entropy = self.state_path_entropy(param, len(data))
        trans_logpdf = self.trans_logpdf(param, data.u[:-1])
        meas_logpdf = self.meas_logpdf(param, data)
        return entropy + trans_logpdf.sum(0) + meas_logpdf.sum(0)

    def elbo_hessian(self, param: Param, data: Data, param_ind=None):
        """Hessian of the ELBO with respect to the parameters."""
        kwd = dict(param_ind=param_ind)
        hess_components = [
            self.state_path_entropy.hessian(param, len(data), **kwd),
            self.trans_logpdf.hessian(param, data.u[:-1], **kwd),
            self.meas_logpdf.hessian(param, data, **kwd),
        ]
        return common.concatenate_coo(*hess_components)
