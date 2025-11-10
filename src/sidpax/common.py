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


def sparse_hessian(f, argnum, vmap_in_axes=None, vmap_out_axes=0):
    """
    Hessian of the sum of a vectorized scalar function `f` in sparse COO format.

    This function returns a callable that computes the Hessian matrix of `f`
    with respect to the specified argument indices (`argnum`), and returns the
    result in sparse COO format suitable for efficient storage and manipulation.

    The returned function is vectorized using JAX's `vmap`, allowing batch
    computation over multiple inputs.

    Parameters
    ----------
    f : Callable
        Scalar function to differentiate. Should accept positional arguments.
    argnum : int or Sequence[int]
        Indices of arguments with respect to which the Hessian is computed.
    vmap_in_axes : None, int or tuple, optional
        Axes specification for vectorization over inputs. Passed as the
        `in_axes` argument to JAX's `vmap`. If `None`, the default, no
        vectorization is applied.
    vmap_out_axes : int or tuple, optional
        Axes specification for vectorization over outputs. Passed as the
        `out_axes` argument to JAX's `vmap`. This argument is ignored if
        `vmap_in_axes` is `None` (default: 0).

    Returns
    -------
    Callable
        A function that computes the sparse Hessian in COO format:
        (values, (row_indices, col_indices)).

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>> from sidpax.common import sparse_hessian, pytree_ind
    >>> import scipy.sparse
    >>> def f(x, y):
    ...     return x**2 + y**3
    >>> x = jnp.array([1.0, 2.0])
    >>> y = jnp.array([3.0, 4.0])
    >>> args = (x, y)
    >>> arginds = pytree_ind(args)
    >>> hess_fn = sparse_hessian(f, argnum=(0, 1), vmap_in_axes=0)
    >>> values, (rows, cols) = hess_fn(args, arginds)
    >>> sparse_hess = scipy.sparse.coo_matrix((values, (rows, cols)))
    >>> argvec, unpack = jax.flatten_util.ravel_pytree(args)
    >>> dense_hess = jax.hessian(lambda argvec: f(*unpack(argvec)).sum())(argvec)
    >>> np.allclose(sparse_hess.todense(), np.array(dense_hess))
    True
    """
    # Ensure argnum is a sequence
    if isinstance(argnum, int):
        argnum = [argnum]

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

    # Return if no vectorization is needed
    if vmap_in_axes is None:
        return wrapped_f

    # Determine the `in_axes` for vmap-ing the wrapped_f
    if isinstance(vmap_in_axes, int):
        wrapper_in_axes = vmap_in_axes
    else:
        argder_in_axes = tuple(
            a for i, a in enumerate(vmap_in_axes) if i in argnum
        )
        wrapper_in_axes = (vmap_in_axes, argder_in_axes)

    # Create new wrapper with vectorization and return
    @functools.wraps(wrapped_f)
    def vmapped(args, arginds):
        coo = jax.vmap(wrapped_f, wrapper_in_axes, vmap_out_axes)(args, arginds)
        return coo[0].flatten(), (coo[1][0].flatten(), coo[1][1].flatten())

    return vmapped


def concatenate_coo(*args):
    """
    Concatenate multiple sparse COO matrices represented as (values, (row, col)).

    This function takes multiple sparse COO matrices, each represented as a 
    triplet of (values, (row_indices, col_indices)), and concatenates them into 
    a single COO matrix. 

    Parameters
    ----------
    *args : tuple
        Variable number of tuples, each containing:
        - values: 1D array of non-zero values.
        - (row_indices, col_indices): Tuple of 1D arrays representing the row and
          column indices of the non-zero values.
    
    Returns
    -------
    tuple
        A tuple representing the concatenated COO matrix in the form:
        (values, (row_indices, col_indices)).
    """
    val = jnp.concatenate([jnp.ravel(h[0]) for h in args])
    row = jnp.concatenate([jnp.ravel(h[1][0]) for h in args])
    col = jnp.concatenate([jnp.ravel(h[1][1]) for h in args])
    return val, (row, col)


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

    return property(getter)
