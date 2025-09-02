"""Unit tests for the `_log_scaling_constant` function."""

from math import isclose, log

import jax.numpy as jnp
import pytest
from jax import Array

from vaxflux._gamma_sum_log_likelihood import _log_scaling_constant


@pytest.mark.parametrize(
    ("alpha", "beta", "expected"),
    [
        (jnp.ones(3), jnp.ones(3), 0.0),
        (jnp.arange(1, 4), jnp.ones(3), 0.0),
        (jnp.ones(3), jnp.arange(1, 4), 2.0 * log(1.0) - log(2.0) - log(3.0)),
    ],
)
def test_scaling_constant_for_select_inputs(
    alpha: Array, beta: Array, expected: float
) -> None:
    """Test the log scaling constant expected value for select inputs."""
    assert len(alpha) == len(beta)
    min_beta = jnp.min(beta)
    log_scaling_const = _log_scaling_constant(alpha, beta, min_beta)
    assert isinstance(log_scaling_const, Array)  # ensure return type is correct
    assert log_scaling_const.shape == ()
    assert isclose(expected, log_scaling_const.item(), rel_tol=1e-6)
