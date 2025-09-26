"""Tests of `sidpax.mat`."""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import sparse

from sidpax import mat


@pytest.fixture(params=range(2))
def rng_seed(request):
    """Pseudo-random number generator seed."""
    return request.param


@pytest.fixture
def rng_key(rng_seed):
    """Jax pseudo-random number generator key"""
    return jax.random.key(rng_seed)


# --- Tests ---


def test_vech_1x1(rng_key):
    M = jax.random.normal(rng_key, (1, 1))
    np.testing.assert_allclose(mat.vech(M), M.flatten())


def test_vech_2x2():
    M = jnp.array([[1, 2], [3, 4]])
    vech_M = jnp.array([1, 3, 4])
    assert all(mat.vech(M) == vech_M)


def test_vech_3x3():
    M = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    vech_M = jnp.array([1, 4, 7, 5, 8, 9])
    assert all(mat.vech(M) == vech_M)
