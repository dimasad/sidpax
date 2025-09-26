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


@pytest.fixture(params=[1, 2, 3, 5])
def n(request):
    """Size of matrix."""
    return request.param


@pytest.fixture(params=[(), (2,), (3, 1, 2)])
def vectorized_shape(request):
    """Shape of matrix vectorization."""
    return request.param


@pytest.fixture
def L(n, rng_key, vectorized_shape: tuple):
    """Random n by n lower-triangular matrix"""
    M = jax.random.normal(rng_key, vectorized_shape + (n, n))
    return jnp.tril(M)


# --- Tests ---


def test_vech_1x1(rng_key):
    M = jax.random.normal(rng_key, (1, 1))
    np.testing.assert_allclose(mat.vech(M), M.flatten(), rtol=0)


def test_vech_2x2():
    M = jnp.array([[1, 2], [3, 4]])
    vech_M = jnp.array([1, 3, 4])
    np.testing.assert_allclose(mat.vech(M), vech_M, rtol=0)


def test_vech_3x3():
    M = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    vech_M = jnp.array([1, 4, 7, 5, 8, 9])
    np.testing.assert_allclose(mat.vech(M), vech_M, rtol=0)


def test_vech_matL(L):
    """Test you recover the same matrix after vech and matl."""
    vech_L = mat.vech(L)
    matl_L = mat.matl(vech_L)
    np.testing.assert_allclose(matl_L, L, rtol=0)


def test_vech_shapes(L, vectorized_shape: tuple):
    """Test matl_size and the shapes of vech."""
    # Test matl_size
    vech_L = mat.vech(L)
    matl_size = mat.matl_size(vech_L.shape[-1])
    assert L.shape[-1] == matl_size

    # Test if the vectorized shape is preserved
    assert vech_L.shape[:-1] == vectorized_shape


def test_matl_diag(L):
    """Test if matl_diag returns the diagonal of the matrix."""
    d = jnp.diagonal(L, axis1=-1, axis2=-2)
    vech_L = mat.vech(L)
    matl_d = mat.matl_diag(vech_L)
    np.testing.assert_allclose(matl_d, d, rtol=0)
