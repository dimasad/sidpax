"""Statistical helper functions."""

import functools
import math

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax_dataclasses as jdc
import numpy as np
from scipy import special

from . import mat


def mvn_logpdf(x, mu, Sigma: mat.PositiveDefiniteMatrix):
    """Log-density of multivariate normal distribution."""
    # Number of dimensions
    n = jnp.shape(x)[-1]

    # log of the normalization factor
    lognfac = -0.5 * n * jnp.log(2 * jnp.pi) - 0.5 * Sigma.logdet

    # Normalized deviations and quadratic sum
    ndev = jsp.linalg.solve_triangular(Sigma.chol_low, x - mu, lower=True)
    quad_sum = -0.5 * jnp.sum(ndev**2, axis=-1)

    return lognfac + quad_sum


@jnp.vectorize
def normal_logpdf_masked(x, mu, std):
    """Normal log-density that returns zero for variates masked with NaN."""
    missing = jnp.isnan(x)
    x_masked = jnp.where(missing, 0.0, x)
    logpdf = jsp.stats.norm.logpdf(x_masked, mu, std)
    return ~missing * logpdf


@jnp.vectorize
def normal_logprob_cdf(x_h, x_l, mu, std):
    """CDF-based Log-probability a normal variable lies within [x_l, x_h]."""
    x_mid = 0.5 * (x_h + x_l)
    logcdf = jnp.r_[
        jsp.stats.norm.logcdf(x_h, mu, std), jsp.stats.norm.logcdf(x_l, mu, std)
    ]
    logsf = jnp.r_[
        jsp.stats.norm.logsf(x_l, mu, std), jsp.stats.norm.logsf(x_h, mu, std)
    ]
    log_summand = jnp.where(x_mid > mu, logcdf, logsf)
    weights = jnp.r_[1, -1]
    return jsp.special.logsumexp(log_summand, b=weights)


@jnp.vectorize
def normal_logprob_trapz(x_h, x_l, mu, std):
    """Trapezoidal Log-probability a normal variable lies within [x_l, x_h]."""
    logpdf = jsp.stats.norm.logpdf(jnp.r_[x_h, x_l], mu, std)
    dx = x_h - x_l
    return jnp.logaddexp(logpdf[0], logpdf[1]) + jnp.log(dx) + jnp.log(0.5)


@jnp.vectorize
def normal_logprob_simps(x_h, x_l, mu, std):
    """Simpson rule Log-probability a normal variable lies within [x_l, x_h]."""
    x_mid = 0.5 * (x_h + x_l)
    logpdf = jsp.stats.norm.logpdf(jnp.r_[x_h, x_mid, x_l], mu, std)
    weights = (x_h - x_l) / 6 * jnp.r_[1, 4, 1]
    return jsp.special.logsumexp(logpdf, b=weights)


@functools.partial(jnp.vectorize, excluded={4})
def normal_logprob_simps_comp(x_h, x_l, mu, std, nsub: int):
    """`normal_logprob_simps` using composite Simpson's rule."""
    x = jnp.linspace(x_l, x_h, 2*nsub + 1)
    logpdf = jsp.stats.norm.logpdf(x, mu, std)
    dx = x_h - x_l
    weights = dx / nsub / 6 * jnp.r_[1, jnp.tile(jnp.r_[4, 2], nsub - 1), 4, 1]
    return jsp.special.logsumexp(logpdf, b=weights)


@functools.partial(jnp.vectorize, excluded={4})
def normal_logprob_simps2_comp(x_h, x_l, mu, std, nsub: int):
    """Composite Simpson's second rule for normal interval log-probability."""
    x = jnp.linspace(x_l, x_h, 3*nsub + 1)
    logpdf = jsp.stats.norm.logpdf(x, mu, std)
    dx = x_h - x_l
    weights = dx / nsub / 8 * jnp.r_[1, jnp.tile(jnp.r_[3, 3, 2], nsub - 1), 3, 3, 1]
    return jsp.special.logsumexp(logpdf, b=weights)

def ghcub(order, dim):
    """Gauss-Hermite nodes and weights for Gaussian cubature."""
    x, w_unnorm = special.roots_hermitenorm(order)
    w = w_unnorm / w_unnorm.sum()
    xrep = [x] * dim
    wrep = [w] * dim
    xmesh = np.meshgrid(*xrep)
    wmesh = np.meshgrid(*wrep)
    X = np.hstack(tuple(xi.reshape(-1, 1) for xi in xmesh))
    W = math.prod(wmesh).flatten()
    return X, W


def sigmapts(dim: int):
    """Sigma points and weights for unscented transform without center point."""
    X = np.r_[np.eye(dim), -np.eye(dim)] * np.sqrt(dim)
    W = 0.5 / dim
    return X, W
