import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import pytest

from sidpax import mat, stats


@pytest.fixture
def x(n, vectorized_shape, rng_key):
    """Random variates."""
    subkey = jax.random.split(rng_key, 3)[1]
    return jax.random.normal(subkey, vectorized_shape + (n,))


@pytest.fixture
def mu(n, vectorized_shape, rng_key):
    """Random mean."""
    subkey = jax.random.split(rng_key, 3)[2]
    return jax.random.normal(subkey, vectorized_shape + (n,))


@pytest.fixture
def Sigma(pos_def_mat):
    """Covariance matrix."""
    return mat.LExpDLT.from_mat(pos_def_mat)


def test_mvn_logpdf(x, mu, Sigma):
    """Compare multivariate logpdf against reference implementation."""
    logpdf = stats.mvn_logpdf(x, mu, Sigma)
    expected = jsp.stats.multivariate_normal.logpdf(x, mu, Sigma.mat)
    np.testing.assert_allclose(logpdf, expected, rtol=1e-10, atol=0)
