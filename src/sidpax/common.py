"""Common functions and utilities."""

import dataclasses
import inspect
import math
from functools import partial
from inspect import signature
from typing import Callable

import hedeut
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt


@hedeut.jax_vectorize(signature="(n,m)->(p)")
def vech(M: npt.ArrayLike) -> npt.NDArray:
    """Pack the lower triangle of a matrix into a vector, columnwise.

    Follows the definition of Magnus and Neudecker (2019), Sec. 3.8,
    DOI: 10.1002/9781119541219
    """
    return M[jnp.triu_indices_from(M.T)[::-1]]


@hedeut.jax_vectorize(signature="(m)->(n,n)")
def matl(v: npt.ArrayLike) -> npt.NDArray:
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


def matl_diag(v: npt.ArrayLike) -> npt.NDArray:
    """Diagonal elements of the entries in the lower triangle a matrix."""
    n = matl_size(len(v))
    i, j = jnp.triu_indices(n)[::-1]
    return v[i == j]


def tria_qr(*args) -> npt.NDArray:
    """Array triangularization routine using QR decomposition."""
    M = jnp.concatenate(args, axis=-1)
    Q, R = jnp.linalg.qr(M.T)
    sig = jnp.sign(jnp.diag(R))
    return R.T * sig


def tria_chol(*args) -> npt.NDArray:
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


@dataclasses.dataclass
class VMapJacobianMethod:
    """vmap-ped array method with sparse Jacobian.

    vmap is done along the first dimension (axis 0) of the arguments listed in
    the set `vec_argnum`. The mapped axis appears in the first dimension
    (axis 0) of the output.

    The sets `vec_argnum` and `jac_argnum` have the indices of the positional
    arguments of the bound method which are vectorized and differentiated,
    respectively.
    """

    f: Callable
    """The underlying method function."""

    vec_argnum: set[int]
    """Indices of the arguments that will be vectorized."""

    jac_argnum: set[int]
    """Indices of the arguments the Jacobian will be differentiated w.r.t."""

    base_out_ndim: int = 1
    """Number of dimensions of the base method, before vmap-ping."""

    class BoundMethod:
        """A method of the outer class bound to a specific object."""

        def __init__(self, method, obj):
            self.method = method
            """The unbound method object."""

            self.obj = obj
            """The object this method is bound to."""

        def __call__(self, *args):
            return self.method(self.obj, *args)

        def jacval(self, *args):
            return self.method.jacval(self.obj, *args)

        def jac_coo(self, args, arginds):
            return self.method.jac_coo(self.obj, args, arginds)

        def deal(self, *arg_objs):
            args = self.method.getargs(*arg_objs)
            return self(*args)

        def deal_jac_coo(self, arg_objs, argind_objs):
            args = self.method.getargs(*arg_objs)
            arginds = self.method.getarginds(*argind_objs)
            return self.jac_coo(args, arginds)

    def __get__(self, obj, objtype=None) -> BoundMethod:
        """Getter for binding this method to an object."""
        return self.BoundMethod(self, obj)

    def getargs(self, *arg_objs):
        argnames = signature(partial(self.f, None)).parameters.keys()
        return getattrs(arg_objs, argnames)

    def getarginds(self, *argind_objs):
        argnames = list(signature(partial(self.f, None)).parameters.keys())
        jacnames = [argnames[i] for i in sorted(self.jac_argnum)]
        return getattrs(argind_objs, jacnames)

    def vmap(self, f: Callable):
        """Vectorize a callable like the underlying bound method."""
        nargs = self.f.__code__.co_argcount - 1
        in_axes = [0 if i in self.vec_argnum else None for i in range(nargs)]
        return jax.vmap(f, in_axes=in_axes, out_axes=0)

    def __call__(self, obj, *args):
        """Bind the underlying method to obj, vmap, and call it."""
        bound_f = self.f.__get__(obj)
        return self.vmap(bound_f)(*args)

    def jacval(self, obj, *args):
        """Bind the underlying method to obj, vmap its Jacobian, and call it."""
        bound_f = self.f.__get__(obj)
        f_jac = jax.jacobian(bound_f, self.jac_argnum)
        return self.vmap(f_jac)(*args)

    def jac_coo(self, obj, args, arginds):
        """Method sparse Jacobian in coo ((i,j), v) format."""
        # Get Jacobian values
        val = self.jacval(obj, *args)

        # Infer output shape and size from Jacobian values
        out_ndim = self.base_out_ndim + 1
        out_shape = jax.tree.leaves(val)[0].shape[:out_ndim]
        out_size = math.prod(out_shape)

        # Get output indices
        out_ind = jnp.arange(out_size).reshape(out_shape)

        # Get the Jacobian row indices by broadcasting the output indices
        def row_ind(val_leaf):
            ndim_diff = val_leaf.ndim - out_ndim
            expanded = jnp.expand_dims(out_ind, -np.arange(ndim_diff) - 1)
            return jnp.broadcast_to(expanded, val_leaf.shape)

        row = jax.tree.map(row_ind, val)

        # Arguments the Jacobian is differentiated w.r.t.
        jac_args = tuple(args[i] for i in sorted(self.jac_argnum))

        # Get the Jacobian column indices by broadcasting the argument indices
        def col_ind(val_leaf, arg_leaf, argind_leaf):
            ndim_diff = val_leaf.ndim - argind_leaf.ndim
            vectorized = out_ndim + jnp.ndim(arg_leaf) > val_leaf.ndim
            expanded = jnp.expand_dims(argind_leaf, np.arange(ndim_diff) + vectorized)
            return jnp.broadcast_to(expanded, val_leaf.shape)

        col = jax.tree.map(col_ind, val, jac_args, tuple(arginds))
        return row, col, val


def vmap_jacobian_method(
    vec_argnum: set[int], jac_argnum: set[int], base_out_ndim: int = 1
) -> Callable[[Callable], VMapJacobianMethod]:
    def decorator(f: Callable) -> VMapJacobianMethod:
        return VMapJacobianMethod(f, vec_argnum, jac_argnum, base_out_ndim)

    return decorator


def getattrs(objs, names):
    """Get a sequence of names from a sequence of objects or dicts."""
    # Initialize return output object
    attrs = []

    # Iterate over names
    for name in names:
        # Initialize found flag for error checking
        found = False
        for obj in objs:
            # Go to next if name found
            if found:
                break

            # Choose getter based on object type
            if isinstance(obj, dict):
                try:
                    attrs.append(obj[name])
                    found = True
                except KeyError:
                    pass
            else:
                try:
                    attrs.append(getattr(obj, name))
                    found = True
                except AttributeError:
                    pass

        # Check for error before proceeding to next name
        if not found:
            raise ValueError(f"Argument '{name}' not found in objs.")
    return tuple(attrs)


def pytree_ind(tree):
    """Map pytree with numeric leaves to their index in flattened array."""
    tree_asint = jax.tree.map(lambda leaf: jnp.astype(leaf, int), tree)
    vector, unpack = jax.flatten_util.ravel_pytree(tree_asint)
    return unpack(jnp.arange(vector.size))
