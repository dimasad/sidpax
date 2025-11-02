"""Classes for representing dynamic system models."""

import collections.abc
import copy
import functools
import inspect
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from sidpax import common


class StateSpaceBase:
    """Base class for state-space model."""

    def bind(self, *args):
        # Make a deep copy of the object
        bound = copy.deepcopy(self)

        for arg in args:
            # Get the mapping of items to bind
            if isinstance(arg, collections.abc.Mapping):
                d = arg
            elif dataclasses.is_dataclass(arg):
                d = dataclasses.asdict(arg)
            else:
                d = arg.__dict__

            # Bind each item in the mapping
            for k, v in d.items():
                setattr(bound, k, v)
        
        return bound
