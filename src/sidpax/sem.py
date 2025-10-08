"""Smoother-Error Method."""

import typing
from dataclasses import dataclass

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
class Estimator:

    model: typing.Any
    """Underlying dynamical system model."""

    Q_diag: bool = False
    """Whether discrete-time process noise covariance Q is diagonal."""

    R_diag: bool = True
    """Whether measurement noise covariance R is diagonal."""

    @jdc.pytree_dataclass
    class Param:
        p: typing.Any
        mu: jax.Array
        Sigma_cond: mat.PositiveDefiniteMatrix
        S_cross: jax.Array

    def param(self, data, rng=None):
        N = len(data)
        p = self.model.param(data, rng)
        mu = jnp.zeros((N, self.model.nx))
        Sigma_cond = stats.LDLTMatrix.I(self.model.nx)
        S_cross = jnp.zeros((self.model.nx, self.model.nx))
        return self.Param(p=p, mu=mu, Sigma_cond=Sigma_cond, S_cross=S_cross)

    def res_cov(self, data, param):
        """Residuals covariance matrices."""
        # Compute residuals
        xres = self.xres(data, param)
        yres = self.yres(data, param)

        # Compute residual outer products
        xres_outer = jnp.einsum("...i,...j->...ij", xres, xres)
        yres_outer = jnp.einsum("...i,...j->...ij", yres, yres)

        # Average over time and posterior sample
        Q = xres_outer.mean((0, 1))
        R = yres_outer.mean((0, 1))

        # Perform Cholesky decomposition and return
        Q_chol = jsp.linalg.cholesky(Q, lower=True)
        R_chol = jsp.linalg.cholesky(R, lower=True)

        # Diagonalize covariances, if needed, and return
        Q_chol = jnp.triu(Q_chol) if self.Q_diag else Q_chol
        R_chol = jnp.triu(R_chol) if self.R_diag else R_chol
        return Q_chol, R_chol

    @staticmethod
    def state_path_entropy(Sigma_cond, N):
        """Differential entropy of the state-path posterior."""
        S_cond = Sigma_cond.chol
        logdet_Sigma_cond = 2 * jnp.log(jnp.diagonal(S_cond)).sum()
        return 0.5 * logdet_Sigma_cond * N

    def cost(self, data, param):
        """Optimization cost function."""
        # Obtain residual covariance matrices
        Q_chol, R_chol = self.res_cov(data, param)

        # Compute covariance log determinant
        logdet_Q = 2 * jnp.log(jnp.diagonal(Q_chol)).sum()
        logdet_R = 2 * jnp.log(jnp.diagonal(R_chol)).sum()

        # Compute average complete data log likelihood
        N = len(data)
        avg_log_like = -0.5 * (N * logdet_R + (N - 1) * logdet_Q)

        # Get the differential entropy of the state-path posterior
        entropy = self.state_path_entropy(param.Sigma_cond, N)

        # Add both terms for the ELBO
        elbo = avg_log_like + entropy

        # Return negate to get the cost for minimization
        return -elbo

    def cost_grad(self, data, param):
        """Cost function gradient."""
        return jax.grad(self.cost, 1)(data, param)

    def xres_cost_hess(self, data, param, param_ind=None):
        """Transition residual cost-function sparse Hessian, in COO format."""
        # Get the parameter indices, if needed
        if param_ind == None:
            param_ind = common.pytree_ind(param)

        # Get the Cholesky factor of the residual covariance matrices
        Q_chol, R_chol = self.res_cov(data, param)

        # Get the transition residual arguments and indices
        args = self.xres_args(data, param)
        args_ind = self.xres_args(data, param_ind)

        # Get the pytree of a single argument block
        u_curr, mu_next, mu_curr = args[:3]
        block_tree = (mu_next[0], mu_curr[0], *args[3:])

        # Ravel (flatten) a block and obtain the unravelling function
        block, unravel_block = ravel_pytree(block_tree)

        # Vectorized function for ravelling flattening all blocks into a matrix
        ravel_blocks = jax.vmap(
            lambda *tree: ravel_pytree(tree)[0],
            in_axes=(0, 0, None, None, None),
        )

        # Equivalent cost function for Hessian, using the squared error
        def sqerr(param_block, u_curr):
            xres = self.xres_sample(u_curr, *unravel_block(param_block))
            xresn = jsp.linalg.solve_triangular(Q_chol, xres.T, lower=True).T
            return 0.5 * jnp.sum(xresn**2, axis=1).mean(axis=0)

        val = jax.vmap(jax.hessian(sqerr))(ravel_blocks(*args[1:]), u_curr)
        row = jnp.repeat(ravel_blocks(*args_ind[1:]), len(block), -1)
        col = jnp.tile(ravel_blocks(*args_ind[1:]), (1, len(block)))
        return val, (row, col)

    def yres_cost_hess(self, data, param, param_ind=None):
        """Output residual cost-function sparse Hessian, in COO format."""
        # Get the parameter indices, if needed
        if param_ind == None:
            param_ind = common.pytree_ind(param)

        # Get the Cholesky factor of the residual covariance matrices
        Q_chol, R_chol = self.res_cov(data, param)

        # Get the pytree of a single argument block
        block_tree = jdc.replace(param, mu=param.mu[0])

        # Ravel (flatten) a block and obtain the unravelling function
        block, unravel_block = ravel_pytree(block_tree)

        # Vectorized function for ravelling flattening all blocks into a matrix
        ravel_blocks = jax.vmap(
            lambda tree: ravel_pytree(tree)[0],
            in_axes=(self.Param(p=None, mu=0, Sigma_cond=None, S_cross=None),),
        )

        # Equivalent cost function for Hessian, using the squared error
        def sqerr(param_block, datum):
            param = unravel_block(param_block)
            yres = self.yres_sample(datum, param)
            yresn = jsp.linalg.solve_triangular(R_chol, yres.T, lower=True).T
            return 0.5 * jnp.sum(yresn**2, axis=1).mean(axis=0)

        val = jax.vmap(jax.hessian(sqerr))(ravel_blocks(param), data)
        row = jnp.repeat(ravel_blocks(param_ind), len(block), -1)
        col = jnp.tile(ravel_blocks(param_ind), (1, len(block)))
        return val, (row, col)

    def cost_hess(self, data, param, param_ind=None):
        xres_hess = self.xres_cost_hess(data, param, param_ind)
        yres_hess = self.yres_cost_hess(data, param, param_ind)
        v = ravel_pytree([h[0] for h in [xres_hess, yres_hess]])[0]
        i = ravel_pytree([h[1][0] for h in [xres_hess, yres_hess]])[0]
        j = ravel_pytree([h[1][1] for h in [xres_hess, yres_hess]])[0]
        return v, (i, j)

    def yres_sample(self, datum: Data, param: Param):
        """Measurement output residuals for a single time sample."""
        # Get the Cholesky factor of the marginal state covariance
        S_cond = param.Sigma_cond.chol
        S_marg = mat.tria_qr(S_cond, param.S_cross)

        # Get the standard normal sigma points and weights.
        # Note that weights will be discarded, as they are all equal.
        std_dev, weights = stats.sigmapts(self.model.nx)

        # Scale the sigma points
        xdev = jnp.matvec(S_marg, std_dev)

        # Add the mean
        xsamp = param.mu + xdev

        # Return the residuals
        return datum.y - self.model.h(xsamp, datum.u, param.p)

    def yres(self, data: Data, param: Param):
        """Measurement output residuals for a full trajectory."""
        in_axes = 0, self.Param(p=None, mu=0, Sigma_cond=None, S_cross=None)
        return jax.vmap(self.yres_sample, in_axes)(data, param)

    def xres_args(self, data, param: Param):
        return (
            data.u[:-1],
            param.mu[1:],
            param.mu[:-1],
            param.p,
            param.Sigma_cond,
            param.S_cross,
        )

    def xres_sample(self, u_curr, mu_next, mu_curr, p, Sigma_cond, S_cross):
        """Discrete-time state transition residuals for a single sample pair."""
        # Get the Cholesky factor of the marginal state covariance
        S_cond = Sigma_cond.chol
        S_marg = mat.tria_qr(S_cond, S_cross)

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
        return x_next - self.model.f(x_curr, u_curr, p)

    def xres(self, data: Data, param: Param):
        """Discrete-time state transition residuals for a full trajectory."""
        in_axes = 0, 0, 0, None, None, None
        xres_sample_vmap = jax.vmap(self.xres_sample, in_axes=in_axes)
        return xres_sample_vmap(*self.xres_args(data, param))
