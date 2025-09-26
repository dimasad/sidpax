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
    """
    return M[jnp.triu_indices_from(M.T)[::-1]]


@hedeut.jax_vectorize(signature="(m)->(n,n)")
def matl(v: ArrayLike) -> jax.Array:
    """Unpack a vector into a square lower triangular matrix."""
    assert v.ndim == 1
    n = matl_size(len(v))
    M = jnp.zeros((n, n))
    return M.at[jnp.triu_indices_from(M)[::-1]].set(v)


def matl_size(vech_len: int) -> int:
    """Number of rows a square matrix `M` given the length of `vech(M)`."""
    n = int(np.sqrt(2 * vech_len + 0.25) - 0.5)
    assert n * (n + 1) / 2 == vech_len
    return n


def matl_diag(v: ArrayLike) -> jax.Array:
    """Diagonal elements of the entries in the lower triangle a matrix."""
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
    def chol(self):
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
    def chol(self):
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
        return cls(jnp.log(jnp.diagonal(mat, -1, -2)))

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
        """The underlying positive-definite matrix."""
        raise NotImplemented

    @classmethod
    def from_mat(cls, mat):
        """Factory that converts a matrix to this representation."""
        raise NotImplemented

    @classmethod
    def identity(cls, shape_or_len):
        """Factory of an identity matrix in this representation."""
        raise NotImplemented


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
    def chol(self):
        """Lower triangular Cholesky factor `L` of the matrix `P = L @ L.T`."""
        log_chol = matl(self.vech_L)
        chol = jsp.linalg.expm(log_chol)
        return chol

    @property
    def mat(self):
        """The underlying positive-definite matrix."""
        chol_T = self.chol.swapaxes(-1, -2)
        return self.chol @ chol_T

    @classmethod
    def from_mat(cls, mat):
        """Factory that converts a matrix to this representation."""
        raise NotImplemented

    @classmethod
    def identity(cls, shape_or_len):
        """Factory of an identity matrix in this representation."""
        if isinstance(shape_or_len, tuple):
            shape = shape_or_len
            if len(shape) < 2:
                raise ValueError("Shape needs at least 2 elements.")
            if shape[-1] != shape[-2]:
                raise ValueError("Shape must be square.")
            return cls(jnp.zeros(shape))
        else:
            n = shape_or_len
            return cls(jnp.zeros((n, n)))


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
    def chol(self):
        """Lower triangular Cholesky factor `L` of the matrix `P = L @ L.T`."""
        return self.L.mat @ self.eD.chol

    @property
    def mat(self):
        """The underlying positive-definite matrix."""
        L = self.L.mat
        LT = jnp.swapaxes(L, -1, -2)
        return L @ self.eD.mat @ LT

    @classmethod
    def from_mat(cls, mat):
        """Factory that converts a matrix to this representation."""
        raise NotImplemented

    @classmethod
    def identity(cls, shape_or_len):
        """Factory of an identity matrix in this representation."""
        raise NotImplemented
