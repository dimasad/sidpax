"""Tests of `sidpax.common`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import sparse

from sidpax import common


def test_vech_1x1(rng_key):
    M = jax.random.normal(rng_key, (1, 1))
    np.testing.assert_allclose(common.vech(M), M.flatten())


def test_vech_2x2():
    M = jnp.array([[1, 2], [3, 4]])
    vech_M = jnp.array([1, 3, 4])
    assert all(common.vech(M) == vech_M)


def test_vech_3x3():
    M = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    vech_M = jnp.array([1, 4, 7, 5, 8, 9])
    assert all(common.vech(M) == vech_M)


def f(x, y, z):
    return 2 * x * jnp.cos(x * y[1] + y[0] + z) + y[1] ** 2 * (z + 1)


@pytest.fixture(params=range(2))
def rng_seed(request):
    """Pseudo-random number generator seed."""
    return request.param


@pytest.fixture
def rng_key(rng_seed):
    """Jax pseudo-random number generator key"""
    return jax.random.key(rng_seed)


@pytest.fixture
def fargs(rng_key):
    """Input argument of f function."""
    keys = jax.random.split(rng_key, 3)
    x = jax.random.normal(keys[0])
    y = jax.random.normal(keys[1], shape=(2,))
    z = jax.random.normal(keys[2])
    return x, y, z


@pytest.fixture(params=[(1, 2), (0, 2), (0, 1)])
def fargnum(request):
    """Argument indices wrt with take the derivative of `f`."""
    return request.param


def test_sparse_hessian_wrt_all(fargs):
    """Test sparse Hessian computation wrt all arguments."""
    # Get problem variables
    arginds = common.pytree_ind(fargs)  # Argument indices for sparse Hessian
    argnum = 0, 1, 2  # Arguments wrt we are differentiating

    # Compute sparse Hessian and convert to COO format
    sparse_hess_fun = common.sparse_hessian(f, argnum)
    sparse_hess = sparse.coo_array(sparse_hess_fun(fargs, arginds))

    # Compute the dense Hessian by ravelling (flattening) the arguments
    argvec, unpack = jax.flatten_util.ravel_pytree(fargs)
    dense_hess = jax.hessian(lambda argvec: f(*unpack(argvec)))(argvec)

    # Assert they are close
    np.testing.assert_allclose(sparse_hess.todense(), dense_hess)


def test_sparse_hessian_wrt_pair(fargs, fargnum):
    """Test sparse Hessian computation wrt a pair of arguments."""
    # Get problem variables
    argder = [a for i, a in enumerate(fargs) if i in fargnum]
    arginds = common.pytree_ind(argder)

    # Compute sparse Hessian and convert to COO format
    sparse_hess_fun = common.sparse_hessian(f, fargnum)
    sparse_hess = sparse.coo_array(sparse_hess_fun(fargs, arginds))

    # Compute the dense Hessian by ravelling (flattening) the arguments
    argvec, unpack = jax.flatten_util.ravel_pytree(fargs)
    dense_hess = jax.hessian(lambda argvec: f(*unpack(argvec)))(argvec)

    # Create a mask to slice the argument pair wrt we differentiate
    mask = np.concatenate(
        [np.repeat(i in fargnum, a.size) for i, a in enumerate(fargs)]
    )

    # Assert they are close
    np.testing.assert_allclose(sparse_hess.todense(), dense_hess[mask][:, mask])
