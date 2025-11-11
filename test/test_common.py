"""Tests of `sidpax.common`."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import sparse

from sidpax import common

# --- Test Functions ---


def simple_add(a, b):
    """A simple function for testing."""
    return a + b


def power(base, exp=2):
    """A function with a default argument."""
    return base**exp


def jax_scalar_func(x, y):
    """A simple scalar function to be vectorized by JAX."""
    return x**2 + y**3


def test_allow_kwargs_basic_functionality():
    """Tests `allow_kwargs` with a simple function."""
    decorated_add = common.allow_kwargs(simple_add)

    # Test with positional arguments
    assert decorated_add(5, 10) == 15, "Should work with positional args"

    # Test with keyword arguments
    assert decorated_add(a=5, b=10) == 15, "Should work with keyword args"
    assert (
        decorated_add(b=10, a=5) == 15
    ), "Should work with reordered keyword args"

    # Test with mixed arguments
    assert decorated_add(5, b=10) == 15, "Should work with mixed args"


def test_allow_kwargs_function_with_defaults():
    """Tests `allow_kwargs` with a function that has default arguments."""
    decorated_power = common.allow_kwargs(power)

    # Test with positional arguments
    assert decorated_power(3) == 9, "Should use default exponent"
    assert (
        decorated_power(3, 3) == 27
    ), "Should override default with positional arg"

    # Test with keyword arguments
    assert decorated_power(base=3) == 9, "Should use default with keyword arg"
    assert (
        decorated_power(base=3, exp=3) == 27
    ), "Should override default with keyword args"
    assert (
        decorated_power(exp=3, base=3) == 27
    ), "Should handle reordered keyword args"


def test_allow_kwargs_invalid_arguments():
    """Tests that `allow_kwargs` raises TypeError for invalid arguments."""
    decorated_add = common.allow_kwargs(simple_add)

    # Too many arguments
    with pytest.raises(TypeError):
        decorated_add(1, 2, 3)

    # Unknown keyword argument
    with pytest.raises(TypeError):
        decorated_add(a=1, c=3)


def test_allow_kwargs_jax_vectorize_integration():
    """
    Tests `allow_kwargs` specific use case with jax.numpy.vectorize.
    """
    # 1. Create the vectorized function
    vectorized_func = jnp.vectorize(jax_scalar_func, signature="(),()->()")

    # 2. Define input data
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([2.0, 3.0, 4.0])

    # 3. Verify that the raw vectorized function fails with keyword arguments
    with pytest.raises(Exception):
        vectorized_func(x=x, y=y)

    # 4. Apply the decorator
    wrapped_vectorized_func = common.allow_kwargs(vectorized_func)

    # 5. Call the wrapped function with keyword arguments and verify it works
    try:
        result_kwargs = wrapped_vectorized_func(x=x, y=y)
        result_reordered = wrapped_vectorized_func(y=y, x=x)
    except TypeError:
        pytest.fail(
            "The decorated function unexpectedly raised a TypeError with kwargs."
        )

    # 6. Calculate expected result and assert correctness
    expected_result = x**2 + y**3
    assert jnp.allclose(result_kwargs, expected_result)
    assert jnp.allclose(result_reordered, expected_result)

    # 7. Ensure it still works with positional args
    result_positional = wrapped_vectorized_func(x, y)
    assert jnp.allclose(result_positional, expected_result)
