"""Smoother-Error Method."""

import typing

import hedeut
import jax
import jax.flatten_util
import jax.numpy as jnp
import jax_dataclasses as jdc

from . import common, stats


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


class Estimator:
    @jdc.pytree_dataclass
    class Param:
        p: typing.Any
        mu: jax.Array
        Sigma_cond: stats.PositiveDefiniteMatrix
        S_cross: jax.Array

    def param(self, data, rng=None):
        N = len(data)
        p = self.model.param(data, rng)
        mu = jnp.zeros((N, self.model.nx))
        Sigma_cond = stats.LDLTMatrix.I(self.model.nx)
        S_cross = jnp.zeros((self.model.nx, self.model.nx))
        return self.Param(p=p, mu=mu, Sigma_cond=Sigma_cond, S_cross=S_cross)

    def __init__(self, model):
        self.model = model

    def extra_args(self, data, param):
        return dict(
            mu_next=param.mu[1:],
            mu_curr=param.mu[:-1],
            u_curr=data.u[:-1],
        )

    def res(self, data, param):
        """All smoother error residuals"""
        extra_args = self.extra_args(data, param)
        outres = self.outres.deal(data, param, extra_args)
        transres = self.transres.deal(data, param, extra_args)
        return outres, transres

    @common.vmap_jacobian_method(
        vec_argnum={0, 1, 2}, jac_argnum={2, 3, 4, 5}, base_out_ndim=2
    )
    def outres(self, y, u, mu, p, Sigma_cond, S_cross):
        """Measurement output residuals."""
        # Get the Cholesky factor of the marginal state covariance
        S_cond = Sigma_cond.chol
        S_marg = common.tria_qr(S_cond, S_cross)

        # Get the standard normal sigma points and weights.
        # Note that weights will be discarded, as they are all equal.
        std_dev, weights = stats.sigmapts(self.model.nx)

        # Scale the sigma points
        xdev = jnp.matvec(S_marg, std_dev)

        # Add the mean
        xsamp = mu + xdev

        # Return the residuals
        return y - self.model.h(xsamp, u, p)

    @common.vmap_jacobian_method(
        vec_argnum={0, 1, 2}, jac_argnum={0, 1, 3, 4, 5}, base_out_ndim=2
    )
    def transres(self, mu_next, mu_curr, u_curr, p, Sigma_cond, S_cross):
        """Discrete-time state transition residuals."""
        # Get the Cholesky factor of the marginal state covariance
        S_cond = Sigma_cond.chol
        S_marg = common.tria_qr(S_cond, S_cross)

        # Get the standard normal sigma points and weights.
        # Note that weights will be discarded, as they are all equal.
        nx = self.model.nx
        std_dev, weights = stats.sigmapts(2 * nx)

        # Scale the sigma points
        x_curr_dev = jnp.matvec(S_marg, std_dev[:, :nx])
        x_next_dev = jnp.matvec(S_cross, std_dev[:, :nx]) + jnp.matvec(
            S_cond, std_dev[:, -nx:]
        )

        # Add the mean
        x_curr = mu_curr + x_curr_dev
        x_next = mu_next + x_next_dev

        # Return the residuals
        return x_next - x_curr - self.model.f(x_curr, u_curr, p)
