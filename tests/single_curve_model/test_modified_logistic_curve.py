import numpy as np
import numpy.typing as npt
import pytest

from vaxflux.single_curve_model import modified_logistic_curve


@pytest.mark.parametrize(
    "t,r,k,c0",
    [
        # Scalar t
        (1.0, 2.0, 0.5, 0.25),
        (np.float64(1.0), np.float64(2.0), np.float64(0.5), np.float64(0.25)),
        (-0.4, 2.0, 0.5, 0.25),
        (np.float64(-0.4), np.float64(2.0), np.float64(0.5), np.float64(0.25)),
        (0.25, 2.0, 0.5, 0.25),
        (np.float64(0.25), np.float64(2.0), np.float64(0.5), np.float64(0.25)),
        (0.5, 0.25, 2.0, 0.0),
        (np.float64(0.5), np.float64(0.25), np.float64(2.0), np.float64(0.0)),
        (-0.5, 0.25, 2.0, 0.0),
        (np.float64(-0.5), np.float64(0.25), np.float64(2.0), np.float64(0.0)),
        (0.0, 0.25, 2.0, 0.0),
        (np.float64(0.0), np.float64(0.25), np.float64(2.0), np.float64(0.0)),
        # Vector t
        (np.linspace(-1.75, 2.25, 100), 2.0, 0.5, 0.25),
        (np.linspace(-3.0, 3.0, 250), 0.25, 2.0, 0.0),
    ],
)
def test_modified_logistic_curve_output_validation(
    t: float | np.float64 | npt.NDArray[np.float64],
    r: float | np.float64,
    k: float | np.float64,
    c0: float | np.float64,
) -> None:
    y = modified_logistic_curve(t, r, k, c0)
    # First check the output type
    if isinstance(t, np.ndarray):
        assert isinstance(y, np.ndarray)
        assert t.dtype == np.float64
    else:
        assert isinstance(y, np.float64)
    # Next check the bounds
    assert np.all(np.greater_equal(y, 0.0))
    assert np.all(np.less_equal(y, k))
    # Check our switch point
    assert np.all(
        np.where(
            np.greater_equal(t, c0),
            np.greater_equal(y, 0.5 * k),
            np.less_equal(y, 0.5 * k),
        )
    )
    # Finally, check that the curve is monotonic
    dt = 0.025 * np.abs(t)
    y1 = modified_logistic_curve(t + dt, r, k, c0)
    assert np.all(np.greater_equal(y1, y))
    y2 = modified_logistic_curve(t - dt, r, k, c0)
    assert np.all(np.less_equal(y2, y2))
