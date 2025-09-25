"""Common functions and utilities."""

import dataclasses
import functools
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


def sparse_hessian(f, argnum):
    @functools.wraps(f)
    def wrapped_f(args, arginds):
        argder = [a for i, a in enumerate(args) if i in argnum]

        vec, unpack = jax.flatten_util.ravel_pytree(argder)
        vecind = jax.flatten_util.ravel_pytree(arginds)[0]

        def fvec(vec):
            arglist = list(args)
            argder = unpack(vec)
            for i, a in zip(argnum, argder):
                arglist[i] = a
            return f(*arglist)
        
        hval = jax.hessian(fvec)(vec).flatten()
        row = jnp.repeat(vecind, len(vec))
        col = jnp.tile(vecind, len(vec))
        return hval, (row, col)
    return wrapped_f


def allow_kwargs(f: Callable):
    """
    Decorator that allows a function that only accepts positional arguments
    to also accept keyword arguments.
    
    This is useful for JAX-transformed functions like jax.diff, jax.hessian,
    and jax.numpy.vectorize which preserve the function signature but only
    accept positional arguments.
    
    Parameters
    ----------
    func : Callable 
        A function that may only accept positional arguments.
        
    Returns
    -------
    A wrapped function that accepts both positional and keyword arguments.
    """
    # Get the signature of the original function
    sig = inspect.signature(f)

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        """
        Wrapper function that accepts both positional and keyword arguments.
        """
        try:
            # Bind the provided arguments (*args, **kwargs) to the
            # function's signature. This checks for correctness and
            # creates an ordered list of arguments.
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
        except TypeError as e:
            # Re-raise TypeError with a more informative message if binding fails
            raise TypeError(
                f"Argument mismatch for {f.__name__}: {e}"
            ) from e

        # Call the original function with the arguments as a positional tuple
        return f(*bound_args.args)

    return wrapper


def jax_vectorize_method(f=None, **kwargs):
    """Decorator for JAX vectorization of a instance method."""
    if f is None:
        return functools.partial(jax_vectorize_method, **kwargs)
    
    # Replace aliases
    if 's' in kwargs:
        kwargs['signature'] = kwargs['s']
        del kwargs['s']
    if 'e' in kwargs:
        kwargs['excluded'] = kwargs['e']
        del kwargs['e']
    
    @functools.wraps(f)
    def getter(obj):
        return allow_kwargs(jax.numpy.vectorize(f.__get__(obj), **kwargs))
    return functools.cached_property(getter)
