"""Global test configuration and shared fixtures."""

import jax
import jax.numpy as jnp
import pytest

# Configure JAX to use 64 bits for all tests to allow use of higher tolerances
jax.config.update("jax_enable_x64", True)


@pytest.fixture(params=range(2))
def rng_seed(request):
    """Pseudo-random number generator seed."""
    return request.param


@pytest.fixture
def rng_key(rng_seed):
    """Jax pseudo-random number generator key"""
    return jax.random.key(rng_seed)


@pytest.fixture(params=[1, 2, 3, 5])
def n(request):
    """Number of dimensions."""
    return request.param


@pytest.fixture(params=[(), (2,), (3, 1, 2)])
def vectorized_shape(request):
    """Shape of matrix vectorization."""
    return request.param


@pytest.fixture
def rand_A(n, rng_key, vectorized_shape: tuple):
    """Random n by n matrix."""
    return jax.random.normal(rng_key, vectorized_shape + (n, n))


@pytest.fixture
def pos_def_mat(rand_A):
    """Positive definite matrix `A @ A.T`"""
    return rand_A @ jnp.swapaxes(rand_A, -1, -2)
