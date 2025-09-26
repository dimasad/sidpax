"""Special Matrices and Matrix Operations."""

import abc

import hedeut
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax_dataclasses as jdc
import numpy as np
from jax.typing import ArrayLike


@hedeut.jax_vectorize(signature="(n,m)->(p)")
def vech(M: ArrayLike) -> jax.Array:
    """Pack the lower triangle of a matrix into a vector, columnwise.

    Follows the definition of Magnus and Neudecker (2019), Sec. 3.8,
    DOI: 10.1002/9781119541219
    
    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>> M = jnp.array([[1, 2], [3, 4]])
    >>> result = vech(M)
    >>> expected = jnp.array([1, 3, 4])
    >>> np.testing.assert_allclose(result, expected, rtol=0)
    
    >>> M = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> result = vech(M)
    >>> expected = jnp.array([1, 4, 7, 5, 8, 9])
    >>> np.testing.assert_allclose(result, expected, rtol=0)
    
    >>> M = jnp.array([[1]])
    >>> result = vech(M)
    >>> expected = jnp.array([1])
    >>> np.testing.assert_allclose(result, expected, rtol=0)
    """
    return M[jnp.triu_indices_from(M.T)[::-1]]


@hedeut.jax_vectorize(signature="(m)->(n,n)")
def matl(v: ArrayLike) -> jax.Array:
    """Unpack a vector into a square lower triangular matrix.
    
    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>> v = jnp.array([1, 3, 4])
    >>> result = matl(v)
    >>> expected = jnp.array([[1, 0], [3, 4]])
    >>> np.testing.assert_allclose(result, expected, rtol=0)
    
    >>> v = jnp.array([1, 4, 7, 5, 8, 9])
    >>> result = matl(v)
    >>> expected = jnp.array([[1, 0, 0], [4, 5, 0], [7, 8, 9]])
    >>> np.testing.assert_allclose(result, expected, rtol=0)
    
    >>> v = jnp.array([1])
    >>> result = matl(v)
    >>> expected = jnp.array([[1]])
    >>> np.testing.assert_allclose(result, expected, rtol=0)
    """
    assert v.ndim == 1
    n = matl_size(len(v))
    M = jnp.zeros((n, n))
    return M.at[jnp.triu_indices_from(M)[::-1]].set(v)


def matl_size(vech_len: int) -> int:
    """Number of rows a square matrix `M` given the length of `vech(M)`.
    
    Examples
    --------
    >>> matl_size(1)
    1
    >>> matl_size(3)
    2
    >>> matl_size(6)
    3
    >>> matl_size(10)
    4
    """
    n = int(np.sqrt(2 * vech_len + 0.25) - 0.5)
    assert n * (n + 1) / 2 == vech_len
    return n


def matl_diag(v: ArrayLike) -> jax.Array:
    """Diagonal elements of the entries in the lower triangle a matrix.
    
    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>> v = jnp.array([1, 3, 4])  # vech of [[1, 0], [3, 4]]
    >>> result = matl_diag(v)
    >>> expected = jnp.array([1, 4])
    >>> np.testing.assert_allclose(result, expected, rtol=0)
    
    >>> v = jnp.array([1, 4, 7, 5, 8, 9])  # vech of [[1, 0, 0], [4, 5, 0], [7, 8, 9]]
    >>> result = matl_diag(v)
    >>> expected = jnp.array([1, 5, 9])
    >>> np.testing.assert_allclose(result, expected, rtol=0)
    
    >>> v = jnp.array([1])  # vech of [[1]]
    >>> result = matl_diag(v)
    >>> expected = jnp.array([1])
    >>> np.testing.assert_allclose(result, expected, rtol=0)
    """
    v = jnp.asarray(v)
    if v.ndim < 1:
        raise ValueError("Input must have at least one dimension.")
    n = matl_size(v.shape[-1])
    i, j = jnp.triu_indices(n)[::-1]
    return v[..., i == j]


def tria_qr(*args) -> jax.Array:
    """Array triangularization routine using QR decomposition."""
    M = jnp.concatenate(args, axis=-1)
    Q, R = jnp.linalg.qr(M.T)
    sig = jnp.sign(jnp.diag(R))
    return R.T * sig


def tria_chol(*args) -> jax.Array:
    """Array triangularization routine using Cholesky decomposition."""
    M = jnp.concatenate(args, axis=-1)
    MMT = M @ M.T
    return jnp.linalg.cholesky(MMT)


@hedeut.jax_vectorize(signature="(k,m),(k,n)->(k,k)")
def tria2_qr(m1, m2):
    """Triangularization of two matrices using QR decomposition."""
    return tria_qr(m1, m2)


@hedeut.jax_vectorize(signature="(k,m),(k,n)->(k,k)")
def tria2_chol(m1, m2):
    """Triangularization of two matrices using Cholesky decomposition."""
    return tria_chol(m1, m2)


@hedeut.jax_vectorize(signature="(n)->(n,n)")
def make_diagonal(d):
    """Make a diagonal array from a vector of elements of its main diagonal."""
    return jnp.diag(d)


class PositiveDefiniteMatrix(abc.ABC):
    """Symmetric positive-definite matrix base class.

    Subclasses can be either a square matrix or an (..., n, n) shaped array of
    matrices, with each PD matrix along the last two dimensions.
    """

    @property
    @abc.abstractmethod
    def logdet(self):
        """Logarithm of the matrix determinant."""

    @property
    @abc.abstractmethod
    def chol_low(self):
        """Lower triangular Cholesky factor `L` of the matrix `P = L @ L.T`."""

    @property
    @abc.abstractmethod
    def mat(self):
        """Return the matrix."""


@jdc.pytree_dataclass
class ExpD(PositiveDefiniteMatrix):
    """
    Positive-definite diagonal matrix represented as $e^D$.

    Only the diagonal elements are stored.
    """

    d: jax.Array
    """Diagonal elements of D."""

    @property
    def logdet(self):
        """Logarithm of the matrix determinant."""
        return self.d.sum(axis=-1)

    @property
    def chol_low(self):
        """Lower triangular Cholesky factor `S` of the matrix `P = S @ S.T`."""
        sqrt_e_d = jnp.exp(0.5 * self.d)
        return make_diagonal(sqrt_e_d)

    @property
    def mat(self):
        """The underlying positive-definite matrix."""
        return make_diagonal(jnp.exp(self.d))

    @classmethod
    def from_mat(cls, mat: ArrayLike):
        """Factory that converts a square matrix to this representation.

        Any off-diagonal elements are discarded, negative diagonal entries
        generate NaN.
        """
        mat = jnp.asarray(mat)
        if mat.ndim < 2:
            raise ValueError("Input must have at least 2 dimensions.")
        if mat.shape[-1] != mat.shape[-2]:
            raise ValueError("Input must be square.")
        return cls(jnp.log(jnp.diagonal(mat, axis1=-2, axis2=-1)))

    @classmethod
    def identity(cls, shape_or_len):
        """Factory of an identity matrix in this representation."""
        if isinstance(shape_or_len, tuple):
            shape = shape_or_len
            if len(shape) < 2:
                raise ValueError("Shape needs at least 2 elements.")
            if shape[-1] != shape[-2]:
                raise ValueError("Shape must be square.")
            return cls(jnp.zeros(shape[:-1]))
        else:
            n = shape_or_len
            return cls(jnp.zeros(n))


@jdc.pytree_dataclass
class LowerUnitriangular:
    """
    Lower Unitriangular matrix, with one on the main diagonal and zero above.

    Only the elements strictly below the main diagonal are stored, with the
    function vech for storing and matl for restoring.
    """

    vech_L: jax.Array
    """Elements below the main diagonal (using function vech) of L."""

    @property
    def mat(self):
        """The underlying unitriangular matrix."""
        n = matl_size(self.vech_L.shape[-1]) + 1  # +1 because vech_L excludes diagonal
        base_shape = self.vech_L.shape[:-1]
        # Create identity matrix
        mat = jnp.zeros(base_shape + (n, n))
        mat = mat.at[..., jnp.arange(n), jnp.arange(n)].set(1.0)  # Set diagonal to 1
        # Set lower triangular part (excluding diagonal)
        if n > 1:
            # Create indices for strictly lower triangular part
            i_indices = []
            j_indices = []
            for i in range(1, n):
                for j in range(i):
                    i_indices.append(i)
                    j_indices.append(j)
            i_indices = jnp.array(i_indices)
            j_indices = jnp.array(j_indices)
            mat = mat.at[..., i_indices, j_indices].set(self.vech_L)
        return mat

    @classmethod
    def from_mat(cls, mat):
        """Factory that converts a matrix to this representation."""
        mat = jnp.asarray(mat)
        if mat.ndim < 2:
            raise ValueError("Input must have at least 2 dimensions.")
        if mat.shape[-1] != mat.shape[-2]:
            raise ValueError("Input must be square.")
        
        n = mat.shape[-1]
        if n == 1:
            # For 1x1 matrices, vech_L is empty
            return cls(jnp.zeros(mat.shape[:-2] + (0,)))
        
        # Extract strictly lower triangular part (excluding diagonal)
        vech_L = []
        for i in range(1, n):
            for j in range(i):
                vech_L.append(mat[..., i, j])
        
        if vech_L:
            vech_L = jnp.stack(vech_L, axis=-1)
        else:
            vech_L = jnp.zeros(mat.shape[:-2] + (0,))
        
        return cls(vech_L)

    @classmethod
    def identity(cls, shape_or_len):
        """Factory of an identity matrix in this representation."""
        if isinstance(shape_or_len, tuple):
            shape = shape_or_len
            if len(shape) < 2:
                raise ValueError("Shape needs at least 2 elements.")
            if shape[-1] != shape[-2]:
                raise ValueError("Shape must be square.")
            n = shape[-1]
            base_shape = shape[:-2]
            # For identity, all off-diagonal elements are zero
            vech_len = n * (n - 1) // 2  # Length for strictly lower triangular
            return cls(jnp.zeros(base_shape + (vech_len,)))
        else:
            n = shape_or_len
            # For identity, all off-diagonal elements are zero
            vech_len = n * (n - 1) // 2  # Length for strictly lower triangular
            return cls(jnp.zeros(vech_len))


@jdc.pytree_dataclass
class ExpLExpLT(PositiveDefiniteMatrix):
    """
    Positive-definite matrix represented as $e^L (e^L)^T$, where L is lower
    triangular and $e^L$ is the matrix exponential.

    Only the elements at and below the main diagonal are stored, with the
    function vech for storing and matl for restoring.
    """

    vech_L: jax.Array
    """Elements at and below the main diagonal (using function vech) of L."""

    @property
    def logdet(self):
        """Logarithm of the matrix determinant."""
        return 2 * matl_diag(self.vech_L).sum(-1)

    @property
    def chol_low(self):
        """Lower triangular Cholesky factor `L` of the matrix `P = L @ L.T`."""
        log_chol = matl(self.vech_L)
        chol_low = jsp.linalg.expm(log_chol)
        return chol_low

    @property
    def mat(self):
        """The underlying positive-definite matrix."""
        chol_T = self.chol_low.swapaxes(-1, -2)
        return self.chol_low @ chol_T

    @classmethod
    def from_mat(cls, mat):
        """Factory that converts a matrix to this representation."""
        mat = jnp.asarray(mat)
        if mat.ndim < 2:
            raise ValueError("Input must have at least 2 dimensions.")
        if mat.shape[-1] != mat.shape[-2]:
            raise ValueError("Input must be square.")
        
        # Compute Cholesky decomposition
        chol_low = jnp.linalg.cholesky(mat)
        
        # Take matrix logarithm of the Cholesky factor
        # Note: funm doesn't handle vectorized inputs, so we use a simpler approach
        # For lower triangular matrices, we can use a direct approach
        if chol_low.ndim == 2:
            log_chol = jnp.real(jsp.linalg.funm(chol_low, jnp.log))
        else:
            # For vectorized inputs, we need to apply funm to each matrix
            def single_logm(mat):
                return jnp.real(jsp.linalg.funm(mat, jnp.log))
            log_chol = jax.vmap(single_logm)(chol_low.reshape(-1, chol_low.shape[-2], chol_low.shape[-1]))
            log_chol = log_chol.reshape(chol_low.shape)
        
        # Pack into vech format
        vech_L = vech(log_chol)
        
        return cls(vech_L)

    @classmethod
    def identity(cls, shape_or_len):
        """Factory of an identity matrix in this representation."""
        if isinstance(shape_or_len, tuple):
            shape = shape_or_len
            if len(shape) < 2:
                raise ValueError("Shape needs at least 2 elements.")
            if shape[-1] != shape[-2]:
                raise ValueError("Shape must be square.")
            n = shape[-1]
            base_shape = shape[:-2]
            vech_len = n * (n + 1) // 2  # Length for lower triangular including diagonal
            return cls(jnp.zeros(base_shape + (vech_len,)))
        else:
            n = shape_or_len
            vech_len = n * (n + 1) // 2  # Length for lower triangular including diagonal
            return cls(jnp.zeros(vech_len))


@jdc.pytree_dataclass
class LExpDLT(PositiveDefiniteMatrix):
    """
    Positive-definite matrix represented as $L(e^D)L^T$, where L is lower
    unitriangular and D is diagonal.

    Uses lower unitriangular and e^D representations internally
    """

    L: LowerUnitriangular
    """Lower unitriangular L matrix."""

    eD: ExpD
    """$e^D$ matrix"""

    @property
    def logdet(self):
        """Logarithm of the matrix determinant."""
        return self.eD.logdet

    @property
    def chol_low(self):
        """Lower triangular Cholesky factor `L` of the matrix `P = L @ L.T`."""
        return self.L.mat @ self.eD.chol_low

    @property
    def mat(self):
        """The underlying positive-definite matrix."""
        L = self.L.mat
        LT = jnp.swapaxes(L, -1, -2)
        return L @ self.eD.mat @ LT

    @classmethod
    def from_mat(cls, mat):
        """Factory that converts a matrix to this representation."""
        mat = jnp.asarray(mat)
        if mat.ndim < 2:
            raise ValueError("Input must have at least 2 dimensions.")
        if mat.shape[-1] != mat.shape[-2]:
            raise ValueError("Input must be square.")
        
        # Use LDL decomposition approach
        # For simplicity, we'll use Cholesky and then decompose
        chol = jnp.linalg.cholesky(mat)
        
        # Extract diagonal part and normalize to get L and D
        # First, get the diagonal elements
        diag_chol = jnp.diagonal(chol, axis1=-1, axis2=-2)
        
        # Create diagonal matrix from Cholesky diagonal
        d = jnp.log(diag_chol ** 2)  # Since chol @ chol.T = mat, diag(chol)^2 = diag(D)
        eD = ExpD(d)
        
        # Get the unit lower triangular part
        # Normalize each row by its diagonal element
        L_mat = chol / diag_chol[..., None, :]
        L = LowerUnitriangular.from_mat(L_mat)
        
        return cls(L=L, eD=eD)

    @classmethod
    def identity(cls, shape_or_len):
        """Factory of an identity matrix in this representation."""
        if isinstance(shape_or_len, tuple):
            shape = shape_or_len
            if len(shape) < 2:
                raise ValueError("Shape needs at least 2 elements.")
            if shape[-1] != shape[-2]:
                raise ValueError("Shape must be square.")
            n = shape[-1]
            base_shape = shape[:-2]
            L = LowerUnitriangular.identity(shape)
            eD = ExpD.identity(shape)
            return cls(L=L, eD=eD)
        else:
            n = shape_or_len
            L = LowerUnitriangular.identity(n)
            eD = ExpD.identity(n)
            return cls(L=L, eD=eD)
