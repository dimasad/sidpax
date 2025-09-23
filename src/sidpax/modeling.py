"""Classes for representing dynamic system models."""

from dataclasses import dataclass
import functools
import inspect
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


class StateSpaceBase:
    """Base class for state-space model."""

    def bind(self, **kwargs):
        return PartialObject(self, **kwargs)


class PartialObject:
    """Object whose methods are bound to specific values."""

    def __init__(self, obj, **kwargs):
        self._obj = obj
        """Underlying object."""

        self._kwargs = kwargs
        """Dictionary of values bound to argument names."""

    def __getattr__(self, name):
        # Get underlying attribute
        attr = getattr(self._obj, name)

        # If `name` is a property, return it
        if not callable(attr):
            return attr
        
        # Get method signature
        method = attr
        sig = inspect.signature(method)

        # Bind kwargs
        for key, val in self._kwargs.items():
            if key in sig.parameters:
                method = functools.partial(method, **{key:val})
        return method
