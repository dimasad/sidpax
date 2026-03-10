"""Helper functions for working with pytrees."""


from typing import Any

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc


@jdc.pytree_dataclass
class MergedPyTree:
    """Sequence of pytrees split into unique and replicated leaves and subtrees.

    The unique subtrees, leaves, and leaf elements are the same for all sequence
    elements. Global is also a good name for them. The replicated subtrees,
    leaves, and leaf elements are replicated for each sequence element (local).
    """

    unique: Any
    """pytree with unique elements or `None` if they are replicated."""

    replicated: list
    """List of pytrees with replicated elements or `None` if they are unique."""

    is_unique: jdc.Static[Any]
    """pytree with `bool` or `jax.Array` of `bool` specifying what is unique."""

    def __getitem__(self, index: int):
        return jax.tree.map(
            leaf_where, self.is_unique, self.unique, self.replicated[index]
        )

    def __len__(self):
        """Number of merged trees."""
        return len(self.replicated)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def leaf_where(condition, x, y):
    if isinstance(condition, jax.Array):
        return jnp.where(condition, x, y)
    else:
        return x if condition else y


def leaf_select(condition, param):
    if isinstance(condition, jax.Array):
        return param[condition.astype(bool)]
    elif condition:
        return param


def leaf_select_not(condition, param):
    if isinstance(condition, jax.Array):
        return param[~condition.astype(bool)]
    elif not condition:
        return param


def leaf_asfloat(condition):
    if isinstance(condition, jax.Array):
        return condition.astype(float)
    else:
        return float(condition)


def pytree_asfloat(condition):
    return jax.tree.map(leaf_asfloat, condition)


def merge_trees(is_unique, *trees) -> MergedPyTree:
    """
    Merges a sequence of pytrees into unique and replicated leaves and subtrees.

    Parameters
    ----------
    is_unique : pytree with `bool` or `jax.Array` of `bool` as leaves.

    *trees : sequence of one or more pytrees
        The trees must have same structure as `is_unique` or have it as a
        prefix.

    Returns
    -------
    merged : MergedPyTree
        Dataclass reprenting the merging, with the unique taken from `trees[0]`.

    Examples
    --------
    >>> from sidpax.sem import merge_trees
    >>> import jax.numpy as jnp
    >>> trees = [
    ...     dict(a=1.0, b=jnp.array([2.0, 3.0]), c=[4.0, 5.0]),
    ...     dict(a=6.0, b=jnp.array([7.0, 8.0]), c=[9.0, 10.0]),
    ... ]
    >>> is_unique = dict(a=True, b=jnp.array([True, False]), c=False)
    >>> merged = merge_trees(is_unique, *trees)

    >>> # unique has only the leaves or subtrees specified in `is_unique`
    >>> merged.unique
    {'a': 1.0, 'b': Array([2.], dtype=...), 'c': None}

    >>> # replicated has each sequence element's non-unique leaves or subtrees
    >>> merged.replicated[0]
    {'a': None, 'b': Array([3.], dtype=...), 'c': [4.0, 5.0]}
    >>> merged.replicated[1]
    {'a': None, 'b': Array([8.], dtype=...), 'c': [9.0, 10.0]}

    >>> # The merged sequence elements can be accessed with indexing
    >>> merged[0]
    {'a': 1.0, 'b': Array([2., 3.], dtype=...), 'c': [4.0, 5.0]}
    """
    if len(trees) == 0:
        raise TypeError("At least one tree required.")

    is_unique_f = pytree_asfloat(is_unique)
    unique = jax.tree.map(leaf_select, is_unique_f, trees[0])
    replicated = [jax.tree.map(leaf_select_not, is_unique_f, t) for t in trees]
    return MergedPyTree(
        unique=unique, replicated=replicated, is_unique=is_unique
    )
