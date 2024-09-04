"""
Functionality for creating a single curve model.

This module contains the needed utilities to create and fit a single curve model. 
Currently exported functionality includes:
- `modified_logistic_curve`
"""

__all__ = ["modified_logistic_curve"]


import numpy as np
import numpy.typing as npt
from scipy.special import expit


def modified_logistic_curve(
    t: float | np.float64 | npt.NDArray[np.float64],
    r: float | np.float64,
    k: float | np.float64,
    c0: float | np.float64,
) -> np.float64 | npt.NDArray[np.float64]:
    """
    A modified and generalized logistic curve.

    Args:
        t: The time to evaluate the logistic curve at.
        r: The growth rate.
        k: The curve max.
        c0: The curve shift point.

    Returns:
        Either a single numpy float or an array of numpy floats depending on the type
        of `t`.

    Examples:
        >>> modified_logistic_curve(1., 2., 0.5, 0.25)
        0.4259764009841553
        >>> import numpy as np
        >>> t = np.linspace(-2.5, 2.5, 6)
        >>> modified_logistic_curve(t, 2., 0.5, 0.25)
        array([0.00261006, 0.01866344, 0.11135007, 0.33958935, 0.46995667,
               0.49571126])
    """
    return k * expit(r * (t - c0))
