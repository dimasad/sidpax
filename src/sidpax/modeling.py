"""Classes for representing dynamic system models."""

import collections.abc
import copy
import functools
import inspect
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax_dataclasses as jdc
import numpy as np

from sidpax import common, mat, stats


class StateSpaceBase:
    """Base class for discrete-time state-space model."""

    nx: int
    """Number of states."""

    nu: int
    """Number of exogenous inputs."""

    ny: int
    """Number of outputs."""

    def bind(self, *args):
        # Make a deep copy of the object
        bound = copy.deepcopy(self)

        for arg in args:
            # Get the mapping of items to bind
            if isinstance(arg, collections.abc.Mapping):
                d = arg
            else:
                d = arg.__dict__

            # Bind each item in the mapping that is not an attribute yet
            for k, v in d.items():
                if getattr(bound, k, None) is None:
                    setattr(bound, k, v)

        return bound

    def prior_logpdf(self, x0):
        """Prior log-density of the initial state and parameters.

        If not overridden, defaults to an improper uniform prior.
        """
        return jnp.array(0.0)

    def free_sim(self, x0, u):
        """Simulate the system without noise."""
        scanfun = lambda x, u: (self.f(x, u), [x, self.h(x, u)])
        carry, (xpath, ypath) = jax.lax.scan(scanfun, x0, u)
        return xpath, ypath


class MVNTransition(StateSpaceBase):
    """Multivariate normal state transition model."""

    Q: mat.PositiveDefiniteMatrix
    """State transition covariance matrix."""

    @common.jax_vectorize_method(signature="(x),(x),(u)->()")
    def trans_logpdf(self, xnext, x, u):
        """Log-density of a state transition, log p(x_{k+1} | x_k, u_k)."""
        return stats.mvn_logpdf(xnext, self.f(x, u), self.Q)


class MVNMeasurement(StateSpaceBase):
    """Multivariate normal measurement model."""

    R: mat.PositiveDefiniteMatrix
    """Measurement covariance matrix."""

    @common.jax_vectorize_method(signature="(y),(x),(u)->()")
    def meas_logpdf(self, y, x, u):
        """Log-density of a measurement, log p(y_k | x_k, u_k)."""
        return stats.mvn_logpdf(y, self.h(x, u), self.R)


class NormalMeasurements(StateSpaceBase):
    """Independent Normal measurement model."""

    y_log_std: jnp.array
    """Logarithm of the measurement scale."""

    @common.jax_vectorize_method(signature="(y),(x),(u)->()")
    def meas_logpdf(self, y, x, u):
        """Log-density of a measurement, log p(y_k | x_k, u_k)."""
        return stats.normal_logpdf_masked(y, self.h(x, u), self.y_std).sum()

    @property
    def y_std(self):
        """Measurement standard deviations."""
        return jnp.exp(self.y_log_std)

    @y_std.setter
    def y_std(self, value):
        self.y_log_std = jnp.log(value)


class EulerDiscretization(StateSpaceBase):
    """Mixin class for Euler discretization of continuous-time dynamics."""

    dt: float
    """Discretization time step."""

    @common.jax_vectorize_method(signature="(x),(u)->(x)")
    def f(self, x, u):
        """Discrete-time state transition function."""
        return x + self.fc(x, u) * self.dt  # Euler's method
