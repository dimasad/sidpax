import jax
import jax.numpy as jnp
import pytest

from sidpax import common


def test_vech_2x2():
    M = jnp.array([[1, 2],
                   [3, 4]])
    vech_M = jnp.array([1, 3, 4])
    assert all(common.vech(M) == vech_M)


def f(x, y):
    return 2 * x * jnp.cos(x*y[1]+y[0]) + y[1]**2


@pytest.fixture(params=range(2))
def rng_seed(request):
    """Pseudo-random number generator seed."""
    return request.param

@pytest.fixture
def rng_key(rng_seed):
    """Jax pseudo-random number generator key"""
    return jax.random.key(rng_seed)

@pytest.fixture
def x(rng_key):
    """`x` input for test functions"""
    return jax.random.normal(rng_key)

@pytest.fixture
def y(rng_key):
    """`y` input for test functions"""
    return jax.random.normal(rng_key, shape=(2,))
