"""Unit tests for the `_gamma_sum_log_likelihood` function."""

from itertools import product
from math import isclose, log
from typing import Final

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from scipy.stats import gamma

from vaxflux._gamma_sum_log_likelihood import _gamma_sum_log_likelihood

RNG: Final[np.random.Generator] = np.random.default_rng(42)
GAMMA_PARAMS: Final[list[float]] = np.geomspace(0.1, 100.0, 10).tolist()
N_SAMPLES: Final[int] = 100_000_000


@pytest.mark.parametrize(
    ("y", "alpha", "beta"),
    [(RNG.gamma(a, scale=b), a, b) for a, b in product(GAMMA_PARAMS, GAMMA_PARAMS)],
)
def test_can_recover_single_gamma_likelihood(
    y: float, alpha: float, beta: float
) -> None:
    """Test that the log likelihood of a single gamma distribution can be recovered."""
    reference_likelihood = gamma.logpdf(y, a=alpha, scale=beta)
    likelihood = _gamma_sum_log_likelihood(y, jnp.array([alpha]), jnp.array([beta]))
    assert isinstance(likelihood, Array)  # ensure return type is correct
    assert isclose(reference_likelihood, likelihood.item(), rel_tol=1e-4)


@pytest.mark.parametrize(
    ("y", "alpha", "beta"),
    [
        (RNG.gamma(sum(a), scale=b), a, b)
        for a, b in product(
            [tuple(RNG.uniform(0.1, 10.0, size=s).tolist()) for s in (2, 3, 5)],
            GAMMA_PARAMS,
        )
    ],
)
def test_can_recover_sum_of_gammas_with_same_scale(
    y: float, alpha: tuple[float, ...], beta: float
) -> None:
    """Test the sum of gammas with the same scale can be recovered."""
    reference_likelihood = gamma.logpdf(y, a=sum(alpha), scale=beta)
    likelihood = _gamma_sum_log_likelihood(
        y, jnp.array(alpha), jnp.array([beta] * len(alpha))
    )
    assert isinstance(likelihood, Array)  # ensure return type is correct
    assert isclose(reference_likelihood, likelihood.item(), rel_tol=1e-4)


@pytest.mark.parametrize(
    ("y", "alpha", "beta"),
    [(3.0, (2.0, 3.0), (4.0, 5.0)), (23.0, (2.0, 3.0), (3.0, 4.0))],
)
def test_can_recover_with_histogram_based_estimate(
    y: float, alpha: tuple[float, ...], beta: tuple[float, ...]
) -> None:
    """Test that the log likelihood can be recovered with a histogram based estimate."""
    samples = np.zeros(N_SAMPLES)
    for a, b in zip(alpha, beta, strict=False):
        samples += RNG.gamma(a, scale=b, size=N_SAMPLES)
    hist, bin_edges = np.histogram(samples, bins=100_000, range=(0, 200), density=True)
    ix = np.argmax(bin_edges > y) - 1
    reference_likelihood = log(hist[ix])
    likelihood = _gamma_sum_log_likelihood(y, jnp.array(alpha), jnp.array(beta))
    assert isinstance(likelihood, Array)
    assert isclose(reference_likelihood, likelihood.item(), rel_tol=6e-1)
