"""Common command-line interface arguments."""

import importlib
import pathlib
from dataclasses import dataclass
from typing import Literal

import jax
import numpy as np
import optax


@dataclass
class JaxArguments:
    """JAX configuration arguments."""

    x64: bool = False
    """Use double precision (64bits) in JAX."""

    platform: Literal["cpu", "gpu"] | None = None
    """JAX platform (processing unit) to use."""

    def __post_init__(self):
        if self.x64:
            jax.config.update("jax_enable_x64", True)
        if self.platform is not None:
            jax.config.update("jax_platform_name", self.platform)


@dataclass
class TestingArguments:
    """Interactive testing configuration arguments."""

    reload: list[str] = []
    """Modules to reload after the script starts."""

    def __post_init__(self):
        """Reload specified modules."""
        for module_name in self.reload:
            module = importlib.import_module(module_name)
            importlib.reload(module)


@dataclass
class RandomArguments:
    """Random number generator configuration arguments."""

    seed: int = 0
    """Random seed for reproducibility."""

    def __post_init__(self):
        """Set the random seed."""
        np.random.seed(self.seed)


@dataclass
class OutputArguments:
    """Output configuration arguments."""

    matout: pathlib.Path | None = None
    """File name to save data in MATLAB format."""

    pickleout: pathlib.Path | None = None
    """Pickle output file."""

    txtout: pathlib.Path | None = None
    """Text output file."""

    paramsout: pathlib.Path | None = None
    """Parameters output file."""


@dataclass
class StochasticOptimizationArguments:
    """Stochastic optimization configuration arguments."""

    lrate0: float = 2e-3
    """Initial learning rate for stochastic optimization."""

    transition_steps: float = 10.0
    """Learning rate transition steps parameter."""

    decay_rate: float = 0.995
    """Learning rate decay rate parameter."""

    epochs: int = 10
    """Number of optimization epochs."""

    display_skip: int = 100
    """Display optimization progress every `n` iterations."""

    @property
    def lrate_sched(self):
        """Learning rate schedule."""
        return optax.exponential_decay(
            self.lrate0, self.transition_steps, self.decay_rate
        )
