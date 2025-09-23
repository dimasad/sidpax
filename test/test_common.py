import jax.numpy as jnp
import pytest

from sidpax import common


def test_vech_2x2():
    M = jnp.array([[1, 2],
                   [3, 4]])
    vech_M = jnp.array([1, 3, 4])
    assert all(common.vech(M) == vech_M)

