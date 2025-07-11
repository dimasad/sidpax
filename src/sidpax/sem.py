"""Smoother-Error Method."""


import typing

import hedeut
import jax
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

    def res(self, data, param):
        y = data.y
        u = data.u
        p = param.p
        mu = param.mu
        Sigma_cond = param.Sigma_cond
        S_cross = param.S_cross
        return self.outres(y,u,mu,p,Sigma_cond, S_cross)

    @hedeut.jax_vectorize_method(s='(y),(u),(x)->(S,y)', e=(3,4,5))
    def outres(self, y, u, mu, p, Sigma_cond, S_cross):
        # Get the Cholesky factor of the marginal state covariance
        S_cond = Sigma_cond.chol
        S_marg = common.tria_qr(S_cond, S_cross)

        # Get the unscaled sigma points
        us_dev, w = stats.sigmapts(self.model.nx)

        # Scale the sigma points
        xdev = jnp.matvec(S_marg, us_dev)

        # Add the mean
        xsamp = mu + xdev

        # Return the residuals
        return y - self.model.h(xsamp, u, p)



