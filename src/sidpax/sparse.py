"""Sparse matrix utilities and sparse objective functions."""

import copy
import functools
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import jax
import jax.flatten_util
import jax.numpy as jnp


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
    >>> from sidpax.sparse import sparse_hessian
    >>> from sidpax.common import pytree_ind
    >>> import scipy.sparse
    >>> # Scalar, vectorized function
    >>> def f(x, y):
    ...     return x**2 + y**3

    >>> # Hessian function arguments
    >>> x = jnp.array([1.0, 2.0])
    >>> y = jnp.array([3.0, 4.0])
    >>> args = (x, y)
    >>> arginds = pytree_ind(args)

    >>> # Create and call Hessian function
    >>> hess_fn = sparse_hessian(f, argnum=(0, 1), vmap_in_axes=0)
    >>> values, (rows, cols) = hess_fn(args, arginds)
    >>> sparse_hess = scipy.sparse.coo_matrix((values, (rows, cols)))

    >>> # Check if it is equivalent to the Hessian of flattened pytree
    >>> vec, unpack = jax.flatten_util.ravel_pytree(args)
    >>> dense_hess = jax.hessian(lambda vec: f(*unpack(vec)).sum())(vec)
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


@dataclass
class SparseObjective:
    """
    A sparse component of the objective function.

    This class represents a component of an objective function that can be
    efficiently computed with sparse Hessian matrices. It supports parameter
    filtering and vectorization using JAX's `vmap`.

    Parameters
    ----------
    fun : Callable
        The underlying objective function.
    param_filter : Callable, optional
        Filters which parameters are used in this component of the objective.
        Default is a lambda function that returns the parameter unchanged.
    vmap_in_axes : None, int or Sequence[Any], optional
        Axes specification for vectorization over inputs. Default is None.
    vmap_out_axes : None, int or Sequence[Any], optional
        Axes specification for vectorization over outputs. Default is 0.
    obj : Any, optional
        The object the objective function is bound to. Default is None.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from sidpax.sparse import SparseObjective
    >>> 
    >>> # Create a simple wrapper class with an objective
    >>> class SimpleModel:
    ...     def simple_obj(self, param, x):
    ...         return jnp.sum((param - x) ** 2)
    >>> 
    >>> # Create a SparseObjective instance
    >>> sparse_obj = SparseObjective(fun=SimpleModel.simple_obj)
    >>> 
    >>> # Bind to model instance
    >>> model = SimpleModel()
    >>> bound_obj = sparse_obj.__get__(model, SimpleModel)
    >>> 
    >>> param = jnp.array([1.0, 2.0])
    >>> x = jnp.array([0.5, 1.5])
    >>> result = bound_obj(param, x)
    >>> float(result)
    0.5
    """

    fun: Callable
    """The underlying objective function."""

    param_filter: Callable = lambda obj, x: x
    """Filters which parameters are used in this component of the objective."""

    vmap_in_axes: None | int | Sequence[Any] = None
    """Axes specification for vectorization over inputs."""

    vmap_out_axes: None | int | Sequence[Any] = 0
    """Axes specification for vectorization over outputs."""

    obj: Any = None
    """The object the objective function is bound to."""

    @property
    def bound_fun(self):
        """
        `fun` bound to the underlying object.

        Returns
        -------
        Callable
            The bound function.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from sidpax.sparse import SparseObjective
        >>> 
        >>> class Model:
        ...     def obj_fun(self, param):
        ...         return jnp.sum(param ** 2)
        >>> 
        >>> model = Model()
        >>> sparse_obj = SparseObjective(fun=Model.obj_fun)
        >>> bound_obj = sparse_obj.__get__(model, Model)
        >>> bound_fun = bound_obj.bound_fun
        >>> result = bound_fun(jnp.array([1.0, 2.0]))
        >>> float(result)
        5.0
        """
        return self.fun.__get__(self.obj, type(self.obj))

    @property
    def bound_param_filter(self):
        """
        `param_filter` bound to the underlying object.

        Returns
        -------
        Callable
            The bound parameter filter function.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from sidpax.sparse import SparseObjective
        >>> 
        >>> class Model:
        ...     def obj_fun(self, param):
        ...         return jnp.sum(param ** 2)
        ...     
        ...     def filter_param(self, param):
        ...         return param[:2]  # Only use first 2 elements
        >>> 
        >>> model = Model()
        >>> sparse_obj = SparseObjective(
        ...     fun=Model.obj_fun,
        ...     param_filter=Model.filter_param
        ... )
        >>> bound_obj = sparse_obj.__get__(model, Model)
        >>> param = jnp.array([1.0, 2.0, 3.0])
        >>> filtered = bound_obj.bound_param_filter(param)
        >>> filtered.shape
        (2,)
        """
        return self.param_filter.__get__(self.obj, type(self.obj))

    def __call__(self, param, *args):
        """
        Call the sparse objective function.

        Parameters
        ----------
        param : Any
            Parameters to the objective function.
        *args : tuple
            Additional arguments to the objective function.

        Returns
        -------
        Any
            The result of the objective function.

        Raises
        ------
        RuntimeError
            If the SparseObjective is not bound to an object.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from sidpax.sparse import SparseObjective
        >>> 
        >>> class Model:
        ...     def obj_fun(self, param, data):
        ...         return jnp.sum((param - data) ** 2)
        >>> 
        >>> model = Model()
        >>> sparse_obj = SparseObjective(fun=Model.obj_fun)
        >>> bound_obj = sparse_obj.__get__(model, Model)
        >>> param = jnp.array([1.0, 2.0])
        >>> data = jnp.array([0.5, 1.5])
        >>> result = bound_obj(param, data)
        >>> float(result)
        0.5
        """
        if self.obj is None:
            raise RuntimeError("Cannot call unbound SparseObjective.")

        # Bind method to object
        fun = self.bound_fun

        # Vectorize if needed
        if self.vmap_in_axes is not None:
            fun = jax.vmap(fun, self.vmap_in_axes, self.vmap_out_axes)

        # Get the function parameters
        fun_param = self.bound_param_filter(param)

        # Call the bound and vectorized method
        return fun(fun_param, *args)

    def __get__(self, obj, objtype=None):
        if obj is None:
            raise TypeError

        self_copy = copy.copy(self)
        self_copy.obj = obj
        return self_copy

    def param_filter_fun(self, param_filter):
        """
        Set the parameter filter function.

        This method allows setting or replacing the parameter filter function
        in a fluent interface style.

        Parameters
        ----------
        param_filter : Callable
            The new parameter filter function.

        Returns
        -------
        SparseObjective
            Returns self for method chaining.

        Examples
        --------
        >>> from sidpax.sparse import SparseObjective
        >>> import jax.numpy as jnp
        >>> 
        >>> def obj_fun(param):
        ...     return jnp.sum(param ** 2)
        >>> 
        >>> def filter_fun(param):
        ...     return param[:2]
        >>> 
        >>> sparse_obj = SparseObjective(fun=obj_fun)
        >>> sparse_obj = sparse_obj.param_filter_fun(filter_fun)
        >>> sparse_obj.param_filter is filter_fun
        True
        """
        self.param_filter = param_filter
        return self

    def hessian(self, param, *args, param_ind=None):
        """
        Compute the sparse Hessian of the objective function.

        Parameters
        ----------
        param : Any
            Parameters to the objective function.
        *args : tuple
            Additional arguments to the objective function.
        param_ind : Any, optional
            Parameter indices for the sparse Hessian. If None, computed
            automatically using `pytree_ind`.

        Returns
        -------
        tuple
            Sparse Hessian in COO format: (values, (row_indices, col_indices)).

        Raises
        ------
        RuntimeError
            If the SparseObjective is not bound to an object.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from sidpax.sparse import SparseObjective
        >>> from sidpax.common import pytree_ind
        >>> 
        >>> class Model:
        ...     def obj_fun(self, param, data):
        ...         return jnp.sum((param - data) ** 2)
        >>> 
        >>> model = Model()
        >>> sparse_obj = SparseObjective(fun=Model.obj_fun)
        >>> bound_obj = sparse_obj.__get__(model, Model)
        >>> param = jnp.array([1.0, 2.0])
        >>> data = jnp.array([0.5, 1.5])
        >>> values, (rows, cols) = bound_obj.hessian(param, data)
        >>> values.shape
        (4,)
        """
        if self.obj is None:
            raise RuntimeError("Hessian requires bound SparseObjective.")

        # Import here to avoid circular dependency
        from . import common

        # Create parameters if needed
        if param_ind is None:
            param_ind = common.pytree_ind(param)

        # Get the function parameters and parameter indices
        fun_param = self.bound_param_filter(param)
        fun_param_ind = self.bound_param_filter(param_ind)

        # Obtain sparse Hessian function
        hess = sparse_hessian(
            self.bound_fun, 0, self.vmap_in_axes, self.vmap_out_axes
        )

        return hess((fun_param, *args), (fun_param_ind,))


def sparse_objective(fun):
    """
    Decorator to create a SparseObjective from a function.

    This is a convenience decorator that wraps a function in a SparseObjective
    instance, preserving the function's metadata.

    Parameters
    ----------
    fun : Callable
        The function to wrap as a SparseObjective.

    Returns
    -------
    SparseObjective
        A SparseObjective instance wrapping the given function.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from sidpax.sparse import sparse_objective
    >>> 
    >>> @sparse_objective
    ... def my_objective(self, param, data):
    ...     return jnp.sum((param - data) ** 2)
    >>> 
    >>> isinstance(my_objective, type(my_objective).__bases__[0])
    True
    """
    return functools.wraps(fun)(SparseObjective(fun))
