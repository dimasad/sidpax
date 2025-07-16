"""Statistical helper functions."""

import abc
import functools
import math
import re
import typing

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax_dataclasses as jdc
import numpy as np
from jax.flatten_util import ravel_pytree
from scipy import special

from . import common


class PositiveDefiniteMatrix(abc.ABC):
    """Positive definite matrix base class."""

    @property
    @abc.abstractmethod
    def logdet(self):
        """Logarithm of the matrix determinant."""

    @property
    @abc.abstractmethod
    def chol(self):
        """Lower triangular Cholesky factor `L` of the matrix `P = L @ L.T`."""

    @abc.abstractmethod
    def __call__(self):
        """Return the positive-definite matrix."""


@jdc.pytree_dataclass
class LogDiagMatrix(PositiveDefiniteMatrix):
    """Diagonal PD matrix represented by its log-diagonal."""

    log_d: jax.Array
    """Logarithm of the diagonal elements."""

    @property
    def logdet(self):
        """Logarithm of the matrix determinant."""
        return self.log_d.sum()

    @property
    def chol(self):
        """Lower triangular Cholesky factor `S` of the matrix `P = S @ S.T`."""
        sqrt_d = jnp.exp(0.5 * self.log_d)
        return jnp.diag(sqrt_d)

    def __call__(self):
        """The underlying positive-definite matrix."""
        return jnp.diag(jnp.exp(self.log_d))


@jdc.pytree_dataclass
class LogCholMatrix(PositiveDefiniteMatrix):
    """PD matrix represented by the matrix logarithm of its Cholesky factor."""

    vech_log_chol: jax.Array
    """Elements at and below the main diagonal (using function vech) of the
    matrix logarithm of the Cholesky factor of a positive definite matrix."""

    @property
    def logdet(self):
        """Logarithm of the matrix determinant."""
        return 2 * common.matl_diag(self.vech_log_chol).sum()

    @property
    def chol(self):
        """Lower triangular Cholesky factor `L` of the matrix `P = L @ L.T`."""
        log_chol = common.matl(self.vech_log_chol)
        chol = jsp.linalg.expm(log_chol)
        return chol

    def __call__(self):
        """The underlying positive-definite matrix."""
        chol_T = self.chol.swapaxes(-1, -2)
        return self.chol @ chol_T


@jdc.pytree_dataclass
class LDLTMatrix(PositiveDefiniteMatrix):
    """PD matrix represented by its LDL^T decomposition.

    The matrix L is unitriangular, and D is diagonal with strictly positive
    entries. Only the elements below the unit diagonal of L are stored. The
    logarithm of the diagonal elements of D are stored.
    """

    vech_L: jax.Array
    """Elements strictly below the main diagonal (using function vech) of L."""

    log_d: jax.Array
    """Logarithm of the diagonal elements."""

    @classmethod
    def I(cls, n: int):
        """Make identity matrix."""
        vech_L = common.vech(jnp.zeros((n - 1, n - 1)))
        log_d = jnp.zeros(n)
        return cls(vech_L=vech_L, log_d=log_d)

    @property
    def logdet(self):
        """Logarithm of the matrix determinant."""
        return self.log_d.sum()

    @property
    def _L(self):
        """The underlying unitriangular matrix `L`."""
        n = self.log_d.shape[-1]
        base_shape = jnp.broadcast_shapes(
            self.log_d.shape[:-1], self.vech_L.shape[:-1])
        L = jnp.zeros(base_shape + (n, n)).at[...].set(jnp.identity(n))
        return L.at[..., 1:, :-1].add(common.matl(self.vech_L))

    @property
    def chol(self):
        """Lower triangular Cholesky factor `S` of the matrix `P = S @ S.T`."""
        sqrt_d = jnp.exp(0.5 * self.log_d)
        return self._L * sqrt_d[..., None, :]

    def __call__(self):
        """The underlying positive-definite matrix."""
        D = jnp.exp(self.log_d[..., None, :])
        L = self._L
        return (L * D) @ L.swapaxes(-1, -2)


def ghcub(order, dim):
    """Gauss-Hermite nodes and weights for Gaussian cubature."""
    x, w_unnorm = special.roots_hermitenorm(order)
    w = w_unnorm / w_unnorm.sum()
    xrep = [x] * dim
    wrep = [w] * dim
    xmesh = np.meshgrid(*xrep)
    wmesh = np.meshgrid(*wrep)
    X = np.hstack(tuple(xi.reshape(-1, 1) for xi in xmesh))
    W = math.prod(wmesh).flatten()
    return X, W


def sigmapts(dim: int):
    """Sigma points and weights for unscented transform without center point."""
    X = np.r_[np.eye(dim), -np.eye(dim)] * np.sqrt(dim)
    W = 0.5 / dim
    return X, W


class SparseResidualFunction:
    def __init__(self, f, in_axes=0, out_axes=0, argnums=0):
        self.f = f
        """The underlying residual method."""

        self.in_axes = in_axes
        """Specification of which input array axes to vmap over."""

        self.out_axes = out_axes
        """"Where the vmap-ed axes should appear in the output."""

        self.argnums = argnums
        """"Argument numbers wrt which the Jacobian is computed."""

    def vfun(self, obj, *args):
        method = self.f.__get__(obj)
        return jax.vmap(method, self.in_axes, self.out_axes)(*args)

    def jacval(self, obj, *args):
        method = self.f.__get__(obj)
        jac = jax.jacobian(method, argnums=self.argnums)
        return jax.vmap(jac, self.in_axes, self.out_axes)(*args)
    
    def jac(self, obj, args, arginds):
        # Get the Jacobian values
        val = self.jacval(obj, *args)

        # Get the residuals to obtain their indices
        r = self.vfun(obj, *args)
        res_ind = np.arange(r.size).reshape(r.shape)

        assert self.out_axes == 0, "This won't work with different axes"

        # Initialize output indices lists
        col = []
        row = []

        # Go over all differentiated arguments and determine column indices
        for n, argval, argind in zip(self.argnums, val, arginds):
            if self.in_axes[n] is None:
                colind = jax.tree.map(
                    lambda i, v: jnp.broadcast_to(i, v.shape), argind, argval
                )
                col.append(colind)
            else:
                extradims = argval.ndim - argind.ndim
                axis = jnp.arange(extradims) + 1
                ind = jnp.expand_dims(argind, axis=axis)
                col.append(jnp.broadcast_to(ind, argval.shape))

        # Go over all differentiated arguments and determine row indices
        for n, argval, argind in zip(self.argnums, val, arginds):
            def treefun(v):
                extradims = v.ndim - res_ind.ndim
                axis = -jnp.arange(extradims) - 1
                return jnp.broadcast_to(jnp.expand_dims(res_ind, axis), v.shape)
            rowind = jax.tree.map(treefun, argval)
            row.append(rowind)

        # Ravel the pytrees and pack in a tuple
        return tuple(ravel_pytree(t)[0] for t in (row, col, val))

    def __get__(self, obj, objtype=None):
        return SparseResidualMethod(self, obj)


class SparseResidualMethod:
    """Sparse residual function bound to an object."""

    def __init__(self, fun: SparseResidualFunction, obj):
        self.fun = fun
        """The underlying sparse residual function."""

        self.obj = obj
        """The object the method is bound to."""

    def __call__(self, *args):
        return self.fun.vfun(self.obj, *args)
    
    def jacval(self, *args):
        return self.fun.jacval(self.obj, *args)

    def jac(self, args, arginds):
        return self.fun.jac(self.obj, args, arginds)


def sparse_residual(f=None, **kwargs):
    if f is None:
        return functools.partial(sparse_residual, **kwargs) 

     # Replace aliases
    if 'i' in kwargs:
        kwargs['in_axes'] = kwargs['i']
        del kwargs['i']
    if 'o' in kwargs:
        kwargs['out_axes'] = kwargs['o']
        del kwargs['o']
    if 'n' in kwargs:
        kwargs['argnums'] = kwargs['n']
        del kwargs['n']

    return SparseResidualFunction(f, **kwargs)
