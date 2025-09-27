"""Common functions and utilities."""

import functools
import inspect
from functools import partial
from inspect import signature
from typing import Callable

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt


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
            raise TypeError(f"Argument mismatch for {f.__name__}: {e}") from e

        # Call the original function with the arguments as a positional tuple
        return f(*bound_args.args)

    return wrapper


def defaultable(func, /, **defaults):
    """
    Like partial, but treat provided args/kwargs as *defaults*:
    callers can override them via positional or keyword arguments.
    """
    sig = inspect.signature(func)
    default_bind = sig.bind_partial(**defaults)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # what the caller actually provided
        call_bind = sig.bind_partial(*args, **kwargs)

        # fill in any missing parameters from our defaults
        for name, val in default_bind.arguments.items():
            if name not in call_bind.arguments:
                call_bind.arguments[name] = val

        # finalize & call
        final_bind = sig.bind(**call_bind.arguments)
        return func(*final_bind.args, **final_bind.kwargs)

    return wrapper


def jax_vectorize_method(f=None, **kwargs):
    """Decorator for JAX vectorization of a instance method."""
    if f is None:
        return functools.partial(jax_vectorize_method, **kwargs)

    # Replace aliases
    if "s" in kwargs:
        kwargs["signature"] = kwargs["s"]
        del kwargs["s"]
    if "e" in kwargs:
        kwargs["excluded"] = kwargs["e"]
        del kwargs["e"]

    @functools.wraps(f)
    def getter(obj):
        return allow_kwargs(jax.numpy.vectorize(f.__get__(obj), **kwargs))

    return functools.cached_property(getter)
