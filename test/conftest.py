"""Global test configuration."""

# Configure JAX to use 64 bits for all tests to allow use of higher tolerances

import jax
jax.config.update("jax_enable_x64", True)

