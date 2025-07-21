"""Smoother-Error Method."""

import typing

import hedeut
import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy as jsp
import jax_dataclasses as jdc
from jax.flatten_util import ravel_pytree

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

    def extra_args(self, data, param: Param):
        return dict(
            mu_next=param.mu[1:],
            mu_curr=param.mu[:-1],
            u_curr=data.u[:-1],
        )

    def res(self, data, param: Param):
        """All smoother error residuals"""
        extra_args = self.extra_args(data, param)
        xres = self.xres.deal(data, param, extra_args)
        yres = self.yres.deal(data, param, extra_args)
        return xres, yres

    def nres(self, data, param):
        """Normalized residuals and their covariance."""
        # Compute residuals
        xres, yres = self.res(data, param)

        # Compute residual outer products
        xres_outer = jnp.einsum("...i,...j->...ij", xres, xres)
        yres_outer = jnp.einsum("...i,...j->...ij", yres, yres)

        # Average over time and posterior sample
        Q = xres_outer.mean((0, 1))
        R = yres_outer.mean((0, 1))

        # Perform Cholesky decomposition
        Q_chol = jsp.linalg.cholesky(Q, lower=True)
        R_chol = jsp.linalg.cholesky(R, lower=True)

        # Normalize residuals and return
        solve_triangular = jnp.vectorize(
            lambda L, b: jsp.linalg.solve_triangular(L, b, lower=True),
            signature="(n,n),(n)->(n)",
        )
        xresn = solve_triangular(Q_chol, xres)
        yresn = solve_triangular(R_chol, yres)
        return xresn, yresn, Q_chol, R_chol

    @staticmethod
    def state_path_entropy(Sigma_cond, N):
        """Differential entropy of the state-path posterior."""
        S_cond = Sigma_cond.chol
        logdet_Sigma_cond = 2 * jnp.log(jnp.diagonal(S_cond)).sum()
        return 0.5 * logdet_Sigma_cond * N

    def cost(self, data, param):
        """Optimization cost function."""
        # Obtain residual covariance matrices
        xresn, yresn, Q_chol, R_chol = self.nres(data, param)

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

    def cost_2step_grad(self, data, param):
        Q_chol, R_chol = self.nres(data, param)[-2:]
        solve_triangular = jnp.vectorize(
            lambda L, b: jsp.linalg.solve_triangular(L, b, lower=True),
            signature="(n,n),(n)->(n)",
        )

        def cost_step2(param):
            xres, yres = self.res(data, param)
            xresn = solve_triangular(Q_chol, xres)
            yresn = solve_triangular(R_chol, yres)

            xsq_err = 0.5 * jnp.sum(xresn**2, (0, 2)).mean()
            ysq_err = 0.5 * jnp.sum(yresn**2, (0, 2)).mean()
            entropy = self.state_path_entropy(param.Sigma_cond, len(data))
            return xsq_err + ysq_err - entropy

        return jax.grad(cost_step2)(param)

    def nres_jac_coo(self, data, param, param_ind=None):
        # Get the parameter indices, if needed
        if param_ind == None:
            param_ind = common.pytree_ind(param)

        extra_args = self.extra_args(data, param)
        extra_arginds = self.extra_args(data, param_ind)

        # Get the residual normalization matrices
        xresn, yresn, Q_chol, R_chol = self.nres(data, param)

        # Invert the residual normalization matrices (please don't judge me)
        Q_chol_inv = jnp.linalg.inv(Q_chol)
        R_chol_inv = jnp.linalg.inv(R_chol)
        Q_inv = Q_chol_inv.T @ Q_chol_inv
        R_inv = R_chol_inv.T @ R_chol_inv

        # Get the residual Jacobian matrices
        xres_jac_coo = self.xres.deal_jac_coo(
            (data, param, extra_args), (param_ind, extra_arginds)
        )
        yres_jac_coo = self.yres.deal_jac_coo((data, param), (param_ind,))

        # Normalize the Jacobian matrices
        xresn_jac_val = jax.tree.map(
            lambda leaf: jnp.einsum("ij,Nsj...->Nsi...", Q_inv, leaf),
            xres_jac_coo[-1],
        )
        yresn_jac_val = jax.tree.map(
            lambda leaf: jnp.einsum("ij,Nsj...->Nsi...", R_inv, leaf),
            yres_jac_coo[-1],
        )

        # Get the number of x residuals and shift row index to stack Jacobians
        nxres = xresn.size
        yres_jac_row = ravel_pytree(yres_jac_coo[0])[0] + nxres

        # Flatten the pytrees and return
        row = ravel_pytree((xres_jac_coo[0], yres_jac_row))[0]
        col = ravel_pytree((xres_jac_coo[1], yres_jac_coo[1]))[0]
        val = ravel_pytree((xresn_jac_val, yresn_jac_val))[0]
        return row, col, val

    def entropy_hess_coo(self, data, param: Param, param_ind=None):
        """Entropy Hessian in COO format, but it will always be zero..."""
        # Get the parameter indices, if needed
        if param_ind == None:
            param_ind = common.pytree_ind(param)

        vec, unpack = ravel_pytree(param.Sigma_cond)
        vec_ind = ravel_pytree(param_ind.Sigma_cond)[0]

        entro_flat = lambda v: self.state_path_entropy(unpack(v), len(data))
        val = jax.hessian(entro_flat)(vec)
        row = jnp.repeat(vec_ind, vec.size)
        col = jnp.tile(vec_ind, vec.size)
        return row, col, val

    @common.vmap_jacobian_method(
        vec_argnum={0, 1, 2}, jac_argnum={2, 3, 4, 5}, base_out_ndim=2
    )
    def yres(self, y, u, mu, p, Sigma_cond, S_cross):
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
        vec_argnum={0, 1, 2},
        jac_argnum={0, 1, 3, 4, 5},
        base_out_ndim=2,
    )
    def xres(self, mu_next, mu_curr, u_curr, p, Sigma_cond, S_cross):
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
