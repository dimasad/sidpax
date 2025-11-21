"""Sparse objective function and Hessian"""

import dataclasses
import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
from scipy.sparse import coo_array


def identity_filter(param):
    """The identity parameter filter, which just returns its argument."""
    return param


@dataclass
class SparseObjective:
    """A sparse component of the objective function.

    The underlying function `fun` is expected to be a method of an object, and
    the `SparseObjective` instance will be bound to that object. The
    `param_filter` is used to select which parameters are used in this component
    of the objective, so the Hessian can be computed with respect to them.
    The Hessian is dense with respect to the filtered parameters, but sparse
    with respect to the full parameter set.

    Before vectorization (or if unvectorized), the function should return a
    scalar `jax.Array`. Vectorization can help computing multiple dense Hessian
    blocks when the full Hessian has a block-sparse structure. The computation
    of the vectorized Hessian assumes all vectorized outputs are summed
    together, so that the multiple Hessian blocks in COO form can just be
    concatenated together.
    """

    fun: Callable
    """The underlying objective function."""

    param_filter: Callable = field(default=staticmethod(identity_filter))
    """Filters which parameters are used in this component of the objective."""

    vmap_in_axes: None | int | Sequence[Any] = None
    """Axes specification for vectorization over the inputs of bound `fun`."""

    vmap_out_axes: None | int | Sequence[Any] = 0
    """Axes specification for vectorization over the outputs of `fun`."""

    obj: None | object = None
    """The object the objective function is bound to (None if unbound)."""

    @property
    def _bound_fun(self):
        """`fun` bound to the underlying object."""
        return self.fun.__get__(self.obj, type(self.obj))

    @property
    def _bound_param_filter(self):
        """`param_filter` bound to the underlying object."""
        return self.param_filter.__get__(self.obj, type(self.obj))

    def __call__(self, param, *args):
        """Binds `fun` to `obj`, vmaps it, filters the first arg, and calls.

        Parameters
        ----------
        param
            The parameters to be filtered and passed to `fun`.
        *args
            Additional arguments to be passed to `fun` after the filtered
            parameters.

        Returns
        -------
        The output of the objective function `fun`.
        """
        if self.obj is None:
            raise RuntimeError("Cannot call unbound SparseObjective.")

        # Bind method to object
        fun = self._bound_fun

        # Vectorize if needed
        if self.vmap_in_axes is not None:
            fun = jax.vmap(fun, self.vmap_in_axes, self.vmap_out_axes)

        # Get the function parameters
        fun_param = self._bound_param_filter(param)

        # Call the bound and vectorized method
        return fun(fun_param, *args)

    def __get__(self, obj, objtype=None):
        """Return a copy of this instance with `obj` set."""
        if obj is None:
            raise TypeError
        return dataclasses.replace(self, obj=obj)

    def set_param_filter(self, param_filter):
        """Sets the parameter filter function, for use as a decorator."""
        return dataclasses.replace(self, param_filter=param_filter)

    def hessian(self, param, *args, param_ind=None):
        """Return the Hessian of `fun` with respect to the filtered params."""
        if self.obj is None:
            raise RuntimeError("Hessian requires bound SparseObjective.")

        # Create parameters if needed
        if param_ind is None:
            param_ind = pytree_ind(param)

        # Get the function parameters and parameter indices
        fun_param = self._bound_param_filter(param)
        fun_param_ind = self._bound_param_filter(param_ind)

        # Obtain sparse Hessian function
        hess = sparse_hessian(
            self._bound_fun, 0, self.vmap_in_axes, self.vmap_out_axes
        )

        return hess((fun_param, *args), (fun_param_ind,))


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
    hess: Callable[[tuple, tuple], tuple]
        A function that computes the sparse Hessian in COO format, the return
        is a tuple in the format (values, (row_indices, col_indices)).

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>> from sidpax.sparse import sparse_hessian, pytree_ind
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


def pytree_ind(tree):
    """Map pytree with numeric leaves to their index in flattened array."""
    tree_asint = jax.tree.map(lambda leaf: jnp.astype(leaf, int), tree)
    vector, unpack = jax.flatten_util.ravel_pytree(tree_asint)
    return unpack(jnp.arange(vector.size))


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
        - (row_indices, col_indices): Tuple of 1D arrays representing the row
          and column indices of the non-zero values.

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


def sparse_objective(fun: Callable) -> SparseObjective:
    """Decorator for creating a `SparseObjective` in a class, from a method.

    Parameters
    ----------
    fun
        The method implementing the objective function that will be wrapped.
        If `fun` is a static or class method, it should be wrapped with the
        appropriate decorator before being passed to this function.

    Examples
    --------

    >>> import jax
    >>> import numpy as np
    >>> import scipy.sparse
    >>> from sidpax.sem import sparse_objective

    >>> # Create a simple problem with a decorated objective function
    >>> class ExampleProblem:
    ...     @sparse_objective
    ...     def obj(self, param_filtered: jax.Array, y: jax.Array):
    ...         return 0.5 * (param_filtered ** 2).sum(0) * y
    ...
    ...     @obj.set_param_filter
    ...     def obj(self, param):
    ...         return param["filtered"]
    ...
    ...     # Vectorize only over first axis of filtered parameters
    ...     obj.vmap_in_axes = (0, None)

    >>> # Create the parameters
    >>> param = dict(filtered=jax.numpy.asarray([[1, 2],[3, 4]], float), z=5.0)
    >>> y = jax.numpy.asarray(2.0)

    >>> # Instantiate the problem and compute the objective
    >>> problem = ExampleProblem()
    >>> obj_value = problem.obj(param, y)
    >>> print(*obj_value)
    5.0 25.0

    >>> # Compute the Hessian
    >>> hess_coo = problem.obj.hessian(param, y)
    >>> hess = scipy.sparse.coo_array(hess_coo).todense()
    >>> np.testing.assert_allclose(hess, np.identity(4) * y)
    """
    return SparseObjective(fun)
