"""
Functionality for creating a single curve model.

This module contains the needed utilities to create and fit a single curve model.
Currently exported functionality includes:
- `modified_logistic_curve`
- `modified_logistic_curve_least_squares`
"""

__all__ = (
    "modified_logistic_curve",
    "modified_logistic_curve_bayes_model",
    "modified_logistic_curve_least_squares",
)


from typing import Any, cast

import arviz as az
import numpy as np
import numpy.typing as npt
import pymc as pm
import pytensor.tensor as pt
from pytensor.compile.ops import as_op
from scipy.optimize import OptimizeResult, least_squares
from scipy.special import expit, logit


def modified_logistic_curve(
    t: float | np.float64 | npt.NDArray[np.float64],
    r: float | np.float64,
    k: float | np.float64,
    c0: float | np.float64,
) -> np.float64 | npt.NDArray[np.float64]:
    """
    Evaluate a modified and generalized logistic curve.

    Args:
        t: The time to evaluate the logistic curve at.
        r: The growth rate.
        k: The curve max.
        c0: The curve shift point.

    Returns:
        Either a single numpy float or an array of numpy floats depending on the type
        of `t`.

    Examples:
        >>> import numpy as np
        >>> from vaxflux.single_curve_model import modified_logistic_curve
        >>> modified_logistic_curve(1.0, 2.0, 0.5, 0.25)
        0.4087872380968218
        >>> t = np.linspace(-2.5, 2.5, 6)
        >>> modified_logistic_curve(t, 2.0, 0.5, 0.25)
        array([0.00203507, 0.01465612, 0.09121276, 0.31122967, 0.46207091,
               0.49450653])

    """
    return k * expit(r * (t - c0))


def modified_logistic_curve_least_squares(
    t: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    x0: tuple[float | None, float | None, float | None] = (None, None, None),
    **kwargs: Any,
) -> OptimizeResult:
    """
    Fit the given data to a modified logistic curve using least squares.

    Args:
        t: The time to evaluate the logistic curve at.
        y: The observed value to fit the logistic curve to.
        x0: A tuple, representing `r`, `k`, and `c0`, of initial parameter guesses for
            the modified logistic curve. If given `None` for any of these this function
            will make an educated guess.
        **kwargs: Further arguments passed to `scipy.optimize.least_squares`.

    Returns:
        The `OptimizeResult` returned from fitting.

    Examples:
        >>> import numpy as np
        >>> from vaxflux.single_curve_model import (
        ...     modified_logistic_curve,
        ...     modified_logistic_curve_least_squares,
        ... )
        >>> rng = np.random.default_rng(seed=123)
        >>> t = np.linspace(-2.0, 2.0, 35)
        >>> y_true = modified_logistic_curve(t, 1.25, 2.5, 0.25)
        >>> y_obs = y_true + rng.normal(scale=0.05)
        >>> opt_result = modified_logistic_curve_least_squares(t, y_obs)
        >>> opt_result.success
        True
        >>> opt_result.x
        array([1.33230644, 2.40095517, 0.25113914])

    """
    # First make educated guesses for the least squares fit
    r0, k0, c0 = x0
    k0 = float(1.1 * np.max(y) if k0 is None else k0)
    c0 = t[np.argsort(y)[len(y) // 2]] if c0 is None else c0
    if r0 is None:
        with np.errstate(divide="ignore"):
            z = logit(y / k0) / (t - c0)
        z = z[np.isfinite(z)]
        r0 = float(np.nanmean(z) if len(z) > 3 else 1.0)
    x0 = (r0, k0, c0)

    # Define the residual function
    def _residuals(x: tuple[float, float, float]) -> npt.NDArray[np.float64]:
        return y - modified_logistic_curve(t, *x)

    # Now fit and return
    opt_result = least_squares(_residuals, x0, **kwargs)  # type: ignore[operator]
    return cast(OptimizeResult, opt_result)


def modified_logistic_curve_bayes_model(
    t: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    sigma: tuple[float, float, float, float] = (3.0, 3.0, 30.0, 1.0),
    **kwargs: dict[str, Any],
) -> tuple[pm.Model, pm.backends.base.MultiTrace | az.InferenceData]:
    """
    Fit the given data to a modified logistic curve using MCMC.

    Args:
        t: The time to evaluate the logistic curve at.
        y: The observed value to fit the logistic curve to.
        sigma: A vector of sigma parameters used in the priors for the model which
            correspond to `r`, `k`, `c0`, and `sigma`.
        **kwargs: Further arguments passed to `pymc.sample`.

    Returns:
        A tuple containing the model itself and the trace from sampling.

    """
    # Least squares fit for initial guesses
    opt_result = modified_logistic_curve_least_squares(t, y)
    r_init, k_init, c0_init = cast(npt.NDArray[np.float64], opt_result.x)

    # Local helpers
    sigma_r, sigma_k, sigma_c0, sigma_sigma = sigma
    t_local = np.copy(t)
    y_local = np.copy(y)

    @as_op(itypes=[pt.dvector], otypes=[pt.dvector])  # type: ignore[no-untyped-call,misc]
    def _modified_logistic_curve(
        theta: tuple[float, ...],
    ) -> np.float64 | npt.NDArray[np.float64]:
        return modified_logistic_curve(t_local, *theta)

    # Construct model
    with pm.Model() as model:
        # Priors
        r = pm.Normal("r", mu=0.0, sigma=sigma_r, initval=r_init)
        k = pm.HalfNormal("k", sigma=sigma_k, initval=k_init)
        c0 = pm.Normal("c0", mu=0.0, sigma=sigma_c0, initval=c0_init)
        sigma = pm.HalfNormal("sigma", sigma=sigma_sigma)

        # Likelihood
        theta = pm.math.stack([r, k, c0])
        pm.Normal(
            "y_obs", mu=_modified_logistic_curve(theta), sigma=sigma, observed=y_local
        )

        # Sample
        trace = pm.sample(**kwargs)

    return model, trace
