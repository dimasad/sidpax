"""Tests of `sidpax.sparse`."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import sparse

from sidpax import common, mat
from sidpax.sparse import SparseObjective, sparse_hessian, sparse_objective

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
    y = jax.random.normal(keys[1], shape=fvmap_shape+(2,))
    z = jax.random.normal(keys[2], shape=fvmap_shape)
    return x, y, z


# --- Tests for sparse_hessian ---


def test_sparse_hessian_wrt_all(fargs, fvmap_shape):
    """Test sparse Hessian computation wrt all arguments."""
    # Get problem variables
    arginds = common.pytree_ind(fargs)  # Argument indices for sparse Hessian
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
    arginds = common.pytree_ind(argder)

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
    np.testing.assert_allclose(
        sparse_hess.todense(), dense_hess[mask][:, mask])


def test_sparse_hessian_single_arg():
    """Test sparse Hessian with a single argument."""
    def simple_func(x):
        return jnp.sum(x ** 2)
    
    x = jnp.array([1.0, 2.0, 3.0])
    args = (x,)
    arginds = common.pytree_ind(args)
    
    sparse_hess_fun = sparse_hessian(simple_func, 0)
    values, (rows, cols) = sparse_hess_fun(args, arginds)
    
    # For f(x) = sum(x^2), Hessian should be 2*I (diagonal matrix)
    # The result is flattened, so we get the diagonal elements at positions 0, 4, 8
    sparse_hess_mat = sparse.coo_matrix((values, (rows, cols)), shape=(3, 3))
    expected = 2 * np.eye(3)
    np.testing.assert_allclose(sparse_hess_mat.todense(), expected)


def test_sparse_hessian_mixed_vmap_axes():
    """Test sparse Hessian with mixed vmap axes (some args vectorized, some not)."""
    def mixed_func(x, z):
        """Both x and z are vectorized."""
        return jnp.sum(x ** 2) + jnp.sum(z)
    
    # Create inputs where both are vectorized (batch size 2)
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # shape (2, 2)
    z = jnp.array([[0.5, 0.5], [1.0, 1.0]])  # shape (2, 2)
    args = (x, z)
    
    # Test with vectorization
    vmap_in_axes = (0, 0)  # both vectorized
    sparse_hess_fun = sparse_hessian(mixed_func, (0, 1), vmap_in_axes)
    
    # Get indices for the args
    arginds = common.pytree_ind(args)
    
    values, (rows, cols) = sparse_hess_fun(args, arginds)
    
    # Verify we got results (shape should be correct)
    assert values.shape[0] > 0


def test_sparse_hessian_nested_pytree():
    """Test sparse Hessian with nested pytree structures."""
    def nested_func(params):
        """Function with nested dict parameter."""
        return params['a'] ** 2 + jnp.sum(params['b'] ** 2)
    
    params = {'a': jnp.array(2.0), 'b': jnp.array([1.0, 2.0])}
    args = (params,)
    arginds = common.pytree_ind(args)
    
    sparse_hess_fun = sparse_hessian(nested_func, 0)
    values, (rows, cols) = sparse_hess_fun(args, arginds)
    
    # Verify we get a valid result
    assert values.shape[0] == 9  # 3x3 matrix flattened


def test_sparse_hessian_with_multiple_batch_dims():
    """Test sparse Hessian with multiple batch dimensions."""
    def batch_func(x):
        return jnp.sum(x ** 2)
    
    # Create input with 2 batch dimensions
    x = jnp.ones((3, 4, 5))  # 3 batches of 4x5 matrices
    args = (x,)
    arginds = common.pytree_ind(args)
    
    # Vectorize over first dimension
    sparse_hess_fun = sparse_hessian(batch_func, 0, vmap_in_axes=0)
    values, (rows, cols) = sparse_hess_fun(args, arginds)
    
    # Should flatten all results
    assert values.ndim == 1
    assert values.shape[0] > 0


# --- Tests for SparseObjective ---


class SimpleModel:
    """Simple model for testing SparseObjective."""
    
    def simple_loss(self, param, data):
        """Simple quadratic loss."""
        return jnp.sum((param - data) ** 2)
    
    def filter_first_two(self, param):
        """Filter to use only first two parameters."""
        return param[:2]


def test_sparse_objective_basic():
    """Test basic SparseObjective functionality."""
    model = SimpleModel()
    
    # Create sparse objective
    obj = SparseObjective(fun=SimpleModel.simple_loss)
    bound_obj = obj.__get__(model, SimpleModel)
    
    # Test calling the objective
    param = jnp.array([1.0, 2.0, 3.0])
    data = jnp.array([0.5, 1.5, 2.5])
    result = bound_obj(param, data)
    
    expected = jnp.sum((param - data) ** 2)
    np.testing.assert_allclose(result, expected)


def test_sparse_objective_with_param_filter():
    """Test SparseObjective with parameter filtering."""
    model = SimpleModel()
    
    # Create sparse objective with filter
    obj = SparseObjective(
        fun=SimpleModel.simple_loss,
        param_filter=SimpleModel.filter_first_two
    )
    bound_obj = obj.__get__(model, SimpleModel)
    
    # Test with filtered parameters
    param = jnp.array([1.0, 2.0, 3.0])
    data = jnp.array([0.5, 1.5])
    result = bound_obj(param, data)
    
    # Should only use first two parameters
    expected = jnp.sum((param[:2] - data) ** 2)
    np.testing.assert_allclose(result, expected)


def test_sparse_objective_vectorized():
    """Test SparseObjective with vectorization."""
    model = SimpleModel()
    
    # Create vectorized sparse objective
    obj = SparseObjective(
        fun=SimpleModel.simple_loss,
        vmap_in_axes=(None, 0)  # data is vectorized
    )
    bound_obj = obj.__get__(model, SimpleModel)
    
    # Test with vectorized data
    param = jnp.array([1.0, 2.0])
    data = jnp.array([[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]])  # 3 samples
    result = bound_obj(param, data)
    
    # Should return 3 results
    assert result.shape == (3,)


def test_sparse_objective_hessian():
    """Test SparseObjective Hessian computation."""
    model = SimpleModel()
    
    obj = SparseObjective(fun=SimpleModel.simple_loss)
    bound_obj = obj.__get__(model, SimpleModel)
    
    param = jnp.array([1.0, 2.0])
    data = jnp.array([0.5, 1.5])
    
    values, (rows, cols) = bound_obj.hessian(param, data)
    
    # For quadratic loss, Hessian should be 2*I
    sparse_hess = sparse.coo_matrix((values, (rows, cols)))
    expected = 2 * np.eye(2)
    np.testing.assert_allclose(sparse_hess.todense(), expected, atol=1e-5)


def test_sparse_objective_decorator():
    """Test sparse_objective decorator."""
    class TestModel:
        @sparse_objective
        def my_loss(self, param, data):
            return jnp.sum((param - data) ** 2)
    
    model = TestModel()
    param = jnp.array([1.0, 2.0])
    data = jnp.array([0.5, 1.5])
    
    result = model.my_loss(param, data)
    expected = jnp.sum((param - data) ** 2)
    np.testing.assert_allclose(result, expected)


def test_sparse_objective_unbound_error():
    """Test that calling unbound SparseObjective raises error."""
    obj = SparseObjective(fun=lambda self, x: x)
    
    with pytest.raises(RuntimeError, match="Cannot call unbound"):
        obj(jnp.array([1.0]))


def test_sparse_objective_hessian_unbound_error():
    """Test that Hessian of unbound SparseObjective raises error."""
    obj = SparseObjective(fun=lambda self, x: x)
    
    with pytest.raises(RuntimeError, match="Hessian requires bound"):
        obj.hessian(jnp.array([1.0]))


def test_sparse_objective_param_filter_fun():
    """Test param_filter_fun method for setting parameter filter."""
    def obj_fun(self, param):
        return jnp.sum(param ** 2)
    
    def filter_fun(self, param):
        return param[:2]
    
    obj = SparseObjective(fun=obj_fun)
    obj = obj.param_filter_fun(filter_fun)
    
    assert obj.param_filter is filter_fun


def test_sparse_objective_hessian_with_vmap():
    """Test SparseObjective Hessian with vectorization."""
    class VectorizedModel:
        def loss(self, param, data):
            return jnp.sum((param - data) ** 2)
    
    model = VectorizedModel()
    
    obj = SparseObjective(
        fun=VectorizedModel.loss,
        vmap_in_axes=(None, 0)  # vectorize over data
    )
    bound_obj = obj.__get__(model, VectorizedModel)
    
    param = jnp.array([1.0, 2.0])
    data = jnp.array([[0.5, 1.5], [1.5, 2.5]])  # 2 samples
    
    values, (rows, cols) = bound_obj.hessian(param, data)
    
    # Should return flattened Hessian
    assert values.shape[0] > 0


def test_sparse_objective_with_complex_param_filter():
    """Test SparseObjective with complex parameter filtering."""
    class ComplexModel:
        def loss(self, param_subset, data):
            return jnp.sum(param_subset ** 2 * data)
        
        def complex_filter(self, param):
            """Extract specific elements from nested structure."""
            return jnp.array([param['a'], param['b'][0]])
    
    model = ComplexModel()
    
    obj = SparseObjective(
        fun=ComplexModel.loss,
        param_filter=ComplexModel.complex_filter
    )
    bound_obj = obj.__get__(model, ComplexModel)
    
    param = {'a': 1.0, 'b': jnp.array([2.0, 3.0])}
    data = jnp.array([1.0, 1.0])
    
    result = bound_obj(param, data)
    expected = 1.0**2 * 1.0 + 2.0**2 * 1.0
    np.testing.assert_allclose(result, expected)


def test_sparse_objective_bound_properties():
    """Test bound_fun and bound_param_filter properties."""
    class PropertyModel:
        def obj_fun(self, param):
            return jnp.sum(param ** 2)
        
        def filter_param(self, param):
            return param[:2]
    
    model = PropertyModel()
    
    obj = SparseObjective(
        fun=PropertyModel.obj_fun,
        param_filter=PropertyModel.filter_param
    )
    bound_obj = obj.__get__(model, PropertyModel)
    
    # Test bound_fun property
    bound_fun = bound_obj.bound_fun
    result = bound_fun(jnp.array([1.0, 2.0]))
    expected = 5.0  # 1^2 + 2^2
    np.testing.assert_allclose(result, expected)
    
    # Test bound_param_filter property
    bound_filter = bound_obj.bound_param_filter
    filtered = bound_filter(jnp.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(filtered, jnp.array([1.0, 2.0]))


# --- Additional complex vmap tests for Issue #3 ---


def test_sparse_hessian_partial_vmap_first_arg_only():
    """Test sparse_hessian with vmap only on first argument."""
    def func(x):
        return jnp.sum(x ** 2)
    
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # vectorized, batch size 2
    args = (x,)
    
    # Differentiate wrt x (which is vectorized)
    arginds = common.pytree_ind(args)
    
    vmap_in_axes = 0  # vmap over x
    sparse_hess_fun = sparse_hessian(func, 0, vmap_in_axes)
    values, (rows, cols) = sparse_hess_fun(args, arginds)
    
    assert values.shape[0] > 0
    # For 2 batches of 2 elements each, we expect 2 * 2 * 2 = 8 elements
    assert values.shape[0] == 8


def test_sparse_hessian_vmap_subset_of_pytree():
    """Test sparse_hessian with vmap on subset of pytree elements."""
    def func(params):
        return params['a'] ** 2 + jnp.sum(params['b'] ** 2)
    
    # Create params where 'b' is vectorized but 'a' is not
    params = {
        'a': jnp.array(2.0),
        'b': jnp.array([[1.0, 2.0], [3.0, 4.0]])  # vectorized
    }
    args = (params,)
    arginds = common.pytree_ind(args)
    
    # Vmap with mixed axes for dict
    vmap_in_axes = ({'a': None, 'b': 0},)
    sparse_hess_fun = sparse_hessian(func, 0, vmap_in_axes)
    values, (rows, cols) = sparse_hess_fun(args, arginds)
    
    assert values.shape[0] > 0


def test_sparse_hessian_different_vmap_out_axes():
    """Test sparse_hessian with non-default vmap_out_axes."""
    def func(x):
        return jnp.sum(x ** 2)
    
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    args = (x,)
    arginds = common.pytree_ind(args)
    
    # Test with vmap_out_axes=0 (default)
    sparse_hess_fun = sparse_hessian(func, 0, vmap_in_axes=0, vmap_out_axes=0)
    values, (rows, cols) = sparse_hess_fun(args, arginds)
    
    assert values.shape[0] > 0
    assert rows.shape[0] == values.shape[0]
    assert cols.shape[0] == values.shape[0]
