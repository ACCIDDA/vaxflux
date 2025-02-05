"""
Functionality for creating a single curve model.

This module contains the needed utilities to create and fit a single curve model. 
Currently exported functionality includes:
- `modified_logistic_curve`
- `modified_logistic_curve_least_squares`
"""

__all__ = (
    "modified_logistic_curve",
    "modified_logistic_curve_least_squares",
    "modified_logistic_curve_bayes_model",
)


from typing import Any, cast

import arviz as az
import numpy as np
import numpy.typing as npt
import pymc as pm
from pytensor.compile.ops import as_op
import pytensor.tensor as pt
from scipy.optimize import OptimizeResult, least_squares
from scipy.special import expit, logit


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
        >>> from vaxflux.single_curve_model import modified_logistic_curve
        >>> import numpy as np
        >>> modified_logistic_curve(1., 2., 0.5, 0.25)
        0.4259764009841553
        >>> t = np.linspace(-2.5, 2.5, 6)
        >>> modified_logistic_curve(t, 2., 0.5, 0.25)
        array([0.00261006, 0.01866344, 0.11135007, 0.33958935, 0.46995667,
               0.49571126])
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
        >>> from vaxflux.single_curve_model import *
        >>> import numpy as np
        >>> rng = np.random.default_rng(seed=123)
        >>> t = np.linspace(-2.0, 2.0, 35)
        >>> y_true = modified_logistic_curve(t, 1.25, 2.5, 0.25)
        >>> y_obs = y_true + rng.normal(scale=0.05)
        >>> opt_result = modified_logistic_curve_least_squares(t, y_obs)
        >>> opt_result
            message: `gtol` termination condition is satisfied.
            success: True
            status: 1
                fun: [-2.178e-02 -1.907e-02 ...  7.354e-03  1.028e-02]
                x: [ 1.332e+00  2.401e+00  2.511e-01]
                cost: 0.0011988388941303575
                jac: [[ 2.443e-01 -4.746e-02  1.446e-01]
                    [ 2.666e-01 -5.507e-02  1.665e-01]
                    ...
                    [-3.593e-01 -8.978e-01  2.934e-01]
                    [-3.393e-01 -9.113e-01  2.585e-01]]
                grad: [-4.267e-09 -6.515e-11 -2.714e-09]
        optimality: 4.2671116235695705e-09
        active_mask: [ 0.000e+00  0.000e+00  0.000e+00]
                nfev: 6
                njev: 6
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
    return opt_result


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
    r_init, k_init, c0_init = cast(npt.NDArray[np.number], opt_result.x)

    # Local helpers
    sigma_r, sigma_k, sigma_c0, sigma_sigma = sigma
    t_local = np.copy(t)
    y_local = np.copy(y)

    @as_op(itypes=[pt.dvector], otypes=[pt.dvector])
    def _modified_logistic_curve(theta):
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
