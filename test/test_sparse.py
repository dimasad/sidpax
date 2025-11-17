"""Tests of `sidpax.sparse`."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import sparse

from sidpax.sparse import sparse_hessian, pytree_ind

# --- Test Functions ---


@functools.partial(jnp.vectorize, signature="(),(2),()->()")
def f(x, y, z):
    """Nonlinear test function with multiple arguments for testing Hessian."""
    return 2 * x * jnp.cos(x * y[1] + y[0] + z) + y[1] ** 2 * (z + 1)


@pytest.fixture(params=[(1, 2), (0, 2), (0, 1)])
def fargnum(request):
    """Argument indices wrt with take the derivative of `f`."""
    return request.param


@pytest.fixture(params=[(), (3,)])
def fvmap_shape(request):
    """Length of vectorization of the arguments of `f`"""
    return request.param


@pytest.fixture
def fargs(rng_key, fvmap_shape):
    """Input argument of f function."""
    keys = jax.random.split(rng_key, 3)
    x = jax.random.normal(keys[0], shape=fvmap_shape)
    y = jax.random.normal(keys[1], shape=fvmap_shape + (2,))
    z = jax.random.normal(keys[2], shape=fvmap_shape)
    return x, y, z


# --- Tests ---


def test_sparse_hessian_wrt_all(fargs, fvmap_shape):
    """Test sparse Hessian computation wrt all arguments."""
    # Get problem variables
    arginds = pytree_ind(fargs)  # Argument indices for sparse Hessian
    argnum = 0, 1, 2  # Arguments wrt we are differentiating

    # Compute sparse Hessian and convert to COO format
    vmap_in_axes = 0 if fvmap_shape else None
    sparse_hess_fun = sparse_hessian(f, argnum, vmap_in_axes)
    sparse_hess = sparse.coo_array(sparse_hess_fun(fargs, arginds))

    # Compute the dense Hessian by ravelling (flattening) the arguments
    argvec, unpack = jax.flatten_util.ravel_pytree(fargs)
    dense_hess = jax.hessian(lambda argvec: f(*unpack(argvec)).sum())(argvec)

    # Assert they are close
    np.testing.assert_allclose(sparse_hess.todense(), dense_hess)


def test_sparse_hessian_wrt_pair(fargs, fargnum, fvmap_shape):
    """Test sparse Hessian computation wrt a pair of arguments."""
    # Get problem variables
    argder = [a for i, a in enumerate(fargs) if i in fargnum]
    arginds = pytree_ind(argder)

    # Compute sparse Hessian and convert to COO format
    vmap_in_axes = 0 if fvmap_shape else None
    sparse_hess_fun = sparse_hessian(f, fargnum, vmap_in_axes)
    sparse_hess = sparse.coo_array(sparse_hess_fun(fargs, arginds))

    # Compute the dense Hessian by ravelling (flattening) the arguments
    argvec, unpack = jax.flatten_util.ravel_pytree(fargs)
    dense_hess = jax.hessian(lambda argvec: f(*unpack(argvec)).sum())(argvec)

    # Create a mask to slice the argument pair wrt we differentiate
    mask = np.concatenate(
        [np.repeat(i in fargnum, a.size) for i, a in enumerate(fargs)]
    )

    # Assert they are close
    np.testing.assert_allclose(sparse_hess.todense(), dense_hess[mask][:, mask])
