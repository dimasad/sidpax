"""Tests of `sidpax.mat`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sidpax import mat


@pytest.fixture(scope="module", autouse=True)
def use_cpu_for_module():
    """Use CPU as the JAX device for all tests in this module."""
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        yield


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


@pytest.fixture
def rand_pos_def_mat(rand_A):
    """Positive definite matrix A @ A.T"""
    return rand_A @ jnp.swapaxes(rand_A, -1, -2)


# --- Tests ---


def test_64bit_enabled_in_tests():
    """Test whether JAX was appropriately configured to use 64 bits."""
    assert jnp.ones(1, dtype=jnp.float64).dtype == jnp.float64


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


# --- Matrix class tests ---


@pytest.mark.parametrize("mat_class", [mat.ExpD, mat.ExpLExpLT, mat.LExpDLT])
def test_identity_matrices_vectorized(mat_class, vectorized_shape, n):
    """Test identity factory with vectorized shapes."""
    shape = vectorized_shape + (n, n)
    identity_rep = mat_class.identity(shape)
    identity_mat = identity_rep.mat
    expected = jnp.broadcast_to(jnp.eye(n), shape)
    np.testing.assert_allclose(identity_mat, expected, rtol=1e-14, atol=1e-16)


def test_exp_d_from_mat(rand_pos_def_mat):
    """Test ExpD factory and roundtrip conversion."""
    # ExpD only works with diagonal matrices, so we take the diagonal
    diag_vals = jnp.diagonal(rand_pos_def_mat, axis1=-1, axis2=-2)
    diag_mat = mat.make_diagonal(diag_vals)

    # Convert to representation
    exp_d = mat.ExpD.from_mat(diag_mat)

    # Check roundtrip
    recovered_mat = exp_d.mat
    np.testing.assert_allclose(recovered_mat, diag_mat, rtol=1e-12, atol=1e-14)

    # Test properties
    expected_logdet = jnp.linalg.slogdet(diag_mat)[1]
    np.testing.assert_allclose(
        exp_d.logdet, expected_logdet, rtol=1e-12, atol=1e-14
    )

    expected_chol = jnp.linalg.cholesky(diag_mat)
    np.testing.assert_allclose(
        exp_d.chol_low, expected_chol, rtol=1e-12, atol=1e-14
    )


def test_exp_l_exp_lt_from_mat(rand_pos_def_mat):
    """Test ExpLExpLT factory and roundtrip conversion."""
    # Convert to representation
    exp_l_exp_lt = mat.ExpLExpLT.from_mat(rand_pos_def_mat)

    # Check roundtrip
    recovered_mat = exp_l_exp_lt.mat
    np.testing.assert_allclose(
        recovered_mat, rand_pos_def_mat, rtol=1e-10, atol=1e-12
    )

    # Test properties
    expected_logdet = jnp.linalg.slogdet(rand_pos_def_mat)[1]
    np.testing.assert_allclose(
        exp_l_exp_lt.logdet, expected_logdet, rtol=1e-10, atol=1e-12
    )

    expected_chol = jnp.linalg.cholesky(rand_pos_def_mat)
    np.testing.assert_allclose(
        exp_l_exp_lt.chol_low, expected_chol, rtol=1e-10, atol=1e-12
    )


def test_l_exp_d_lt_from_mat(rand_pos_def_mat):
    """Test LExpDLT factory and roundtrip conversion."""
    # Convert to representation
    l_exp_d_lt = mat.LExpDLT.from_mat(rand_pos_def_mat)

    # Check roundtrip
    recovered_mat = l_exp_d_lt.mat
    np.testing.assert_allclose(
        recovered_mat, rand_pos_def_mat, rtol=1e-8, atol=1e-10
    )

    # Test properties
    expected_logdet = jnp.linalg.slogdet(rand_pos_def_mat)[1]
    np.testing.assert_allclose(
        l_exp_d_lt.logdet, expected_logdet, rtol=1e-8, atol=1e-10
    )

    expected_chol = jnp.linalg.cholesky(rand_pos_def_mat)
    np.testing.assert_allclose(
        l_exp_d_lt.chol_low, expected_chol, rtol=1e-8, atol=1e-10
    )


def test_lower_unitriangular_identity(n):
    """Test LowerUnitriangular identity."""
    identity_rep = mat.LowerUnitriangular.identity(n)
    identity_mat = identity_rep.mat
    expected = jnp.eye(n)
    np.testing.assert_allclose(identity_mat, expected, rtol=0)


def test_lower_unitriangular_from_mat():
    """Test LowerUnitriangular from_mat with known matrix."""
    # Create a known lower unitriangular matrix
    test_mat = jnp.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [-0.3, 0.2, 1.0]])

    # Convert to representation
    lower_unit = mat.LowerUnitriangular.from_mat(test_mat)

    # Check roundtrip
    recovered_mat = lower_unit.mat
    np.testing.assert_allclose(recovered_mat, test_mat, rtol=1e-12, atol=1e-14)
