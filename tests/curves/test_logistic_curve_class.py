"""Unit tests for the `vaxflux.curves.LogisticCurve` class."""

from typing import Final

import numpy as np
import numpy.typing as npt
import pytest

from vaxflux.curves import LogisticCurve

PERCENT_BOUND: Final[float] = 150.0


@pytest.mark.parametrize(
    "t",
    [
        np.arange(0.0, 100.0, 10.0),
        np.arange(-100.0, 100.0, 25.0),
        np.arange(-15.0, 225.0, 37.0),
    ],
)
@pytest.mark.parametrize("m", [-0.25, 0.25, 1.25])
@pytest.mark.parametrize("r", [-4.0, -1.0, 0.1])
@pytest.mark.parametrize("s", [10.0, 25.0, 75.0])
def test_propose_p0_within_fixed_error_bound(
    t: npt.NDArray[np.float64], m: float, r: float, s: float
) -> None:
    """Test that `propose_p0` returns a value within 15% of the initial p0."""
    curve = LogisticCurve()
    y = curve.prevalence(t, m=m, r=r, s=s).eval()
    m0, r0, s0 = curve.propose_p0(t, y)
    assert abs(m0 - m) / (0.5 * abs(m0 + m)) <= PERCENT_BOUND
    assert abs(r0 - r) / (0.5 * abs(r0 + r)) <= PERCENT_BOUND
    assert abs(s0 - s) / (0.5 * abs(s0 + s)) <= PERCENT_BOUND
