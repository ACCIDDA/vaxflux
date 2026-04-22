__all__: list[str] = []

import inspect
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

from vaxflux._plot import _plot_curve_figure
from vaxflux._types import NumericalArrayLike
from vaxflux._util import _numerical_array_like_to_1d_jax_array

if TYPE_CHECKING:
    from matplotlib.figure import Figure
else:
    Figure = Any  # type: ignore[misc,assignment]


def _normalize_parameter_sets(
    *,
    parameters: tuple[str, ...],
    parameter_sets: Sequence[dict[str, NumericalArrayLike]] | None,
    kwargs: dict[str, NumericalArrayLike],
) -> list[dict[str, NumericalArrayLike]]:
    """
    Return normalized parameter sets for plotting.

    Args:
        parameters: The parameter names required by the curve.
        parameter_sets: Optional collection of parameter dictionaries to overlay on the
            same axes. When omitted, `kwargs` are used as a single set.
        kwargs: Additional parameters required by the curve methods when plotting a
            single parameter set.

    Returns:
        A list of parameter dictionaries for plotting.

    Raises:
        ValueError: If neither `parameter_sets` nor `kwargs` are provided.
        ValueError: If `parameter_sets` and `kwargs` are both provided.
        ValueError: If any normalized parameter set does not match `parameters`.

    Examples:
        >>> from vaxflux._curves import _normalize_parameter_sets
        >>> _normalize_parameter_sets(
        ...     parameters=("m", "r", "s"),
        ...     parameter_sets=[
        ...         {"m": 0.5, "r": 1.0, "s": 1.0},
        ...         {"m": 0.8, "r": 1.5, "s": 0.5},
        ...     ],
        ...     kwargs={},
        ... )
        [{'m': 0.5, 'r': 1.0, 's': 1.0}, {'m': 0.8, 'r': 1.5, 's': 0.5}]
        >>> _normalize_parameter_sets(
        ...     parameters=("m", "r", "s"),
        ...     parameter_sets=None,
        ...     kwargs={"m": 0.5, "r": 1.0, "s": 1.0},
        ... )
        [{'m': 0.5, 'r': 1.0, 's': 1.0}]
        >>> _normalize_parameter_sets(
        ...     parameters=("m", "r", "s"),
        ...     parameter_sets=[{"m": 0.5, "r": 1.0, "s": 1.0}],
        ...     kwargs={"m": 0.8, "r": 1.5, "s": 0.5},
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Pass either `parameter_sets` or `kwargs`, but not both.
        >>> _normalize_parameter_sets(
        ...     parameters=("m", "r", "s"),
        ...     parameter_sets=None,
        ...     kwargs={},
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Provide either `parameter_sets` or `kwargs`.
        >>> _normalize_parameter_sets(
        ...     parameters=("m", "r", "s"),
        ...     parameter_sets=[{"m": 0.5, "r": 1.0}],
        ...     kwargs={},
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Parameter set keys must exactly match `parameters`: ('m', 'r', 's').
    """
    if parameter_sets is None and not kwargs:
        msg = "Provide either `parameter_sets` or `kwargs`."
        raise ValueError(msg)
    if parameter_sets is not None and kwargs:
        msg = "Pass either `parameter_sets` or `kwargs`, but not both."
        raise ValueError(msg)
    normalized_parameter_sets = (
        [dict(kwargs)]
        if parameter_sets is None
        else [dict(params) for params in parameter_sets]
    )
    if not normalized_parameter_sets:
        msg = "Provide either `parameter_sets` or `kwargs`."
        raise ValueError(msg)
    expected_keys = set(parameters)
    for params in normalized_parameter_sets:
        if set(params) != expected_keys:
            msg = f"Parameter set keys must exactly match `parameters`: {parameters}."
            raise ValueError(msg)
    return normalized_parameter_sets


class Curve(ABC):
    """
    Abstract class for implementations of uptake curves.

    Attributes:
        parameters: A tuple of parameter names required by the prevalence function.

    Examples:
        >>> import jax.numpy as jnp
        >>> from vaxflux import Curve
        >>> class ClampedSlopeCurve(Curve):
        ...     def prevalence(
        ...         self, t: NumericalArrayLike, m: NumericalArrayLike
        ...     ) -> jax.Array:
        ...         return jnp.clip(t * m, 0.0, 1.0)
        >>> t = jnp.linspace(-1.0, 3.0, num=8)
        >>> curve = ClampedSlopeCurve()
        >>> curve.parameters
        ('m',)
        >>> curve.prevalence(t, m=0.5)
        Array([0.        , 0.        , 0.0714286 , 0.35714293, 0.6428572 ,
               0.9285715 , 1.        , 1.        ], dtype=float32)
        >>> curve.incidence(t, m=0.5)
        Array([0. , 0. , 0.5, 0.5, 0.5, 0.5, 0. , 0. ], dtype=float32)

    """

    parameters: tuple[str, ...]

    def __init__(self) -> None:
        self.parameters = tuple(inspect.signature(self.prevalence).parameters)[1:]
        in_axes = (0,) + (None,) * len(self.parameters)
        self._grad_prevalence = jax.vmap(jax.grad(self.prevalence), in_axes=in_axes)

    @abstractmethod
    def prevalence(
        self, t: NumericalArrayLike, **kwargs: NumericalArrayLike
    ) -> jax.Array:
        """
        Compute the prevalence at time `t`.

        Args:
            t: The time steps to evaluate the prevalence curve at.
            **kwargs: Additional parameters required by the prevalence model.

        Returns:
            The prevalence curve evaluated at time `t` bounded between 0 and 1.
        """
        raise NotImplementedError

    def incidence(
        self, t: NumericalArrayLike, **kwargs: NumericalArrayLike
    ) -> jax.Array:
        """
        Compute the incidence at time `t`.

        Args:
            t: The time steps to evaluate the incidence curve at.
            **kwargs: Additional parameters required by the incidence model.

        Returns:
            The incidence curve evaluated at time `t`.
        """
        return cast("jax.Array", self._grad_prevalence(*((t, *tuple(kwargs.values())))))

    def prevalence_difference(
        self,
        t0: NumericalArrayLike,
        t1: NumericalArrayLike,
        **kwargs: NumericalArrayLike,
    ) -> jax.Array:
        """
        Compute prevalence differences between two time arrays.

        Args:
            t0: The start time steps for the interval.
            t1: The end time steps for the interval.
            **kwargs: Additional parameters required by the prevalence model.

        Returns:
            The prevalence difference for each interval.
        """
        return self.prevalence(jnp.asarray(t1), **kwargs) - self.prevalence(
            jnp.asarray(t0), **kwargs
        )

    def plot(
        self,
        t: NumericalArrayLike,
        *,
        parameter_sets: Sequence[dict[str, NumericalArrayLike]] | None = None,
        labels: Sequence[str] | None = None,
        plot_incidence: bool = True,
        plot_prevalence: bool = True,
        figsize: tuple[float, float] | None = None,
        title: str | None = None,
        **kwargs: NumericalArrayLike,
    ) -> Figure:
        """
        Plot the curve's incidence and/or prevalence over time.

        Args:
            t: The time steps to evaluate and plot.
            parameter_sets: Optional collection of parameter dictionaries to overlay on
                the same axes. When omitted, `kwargs` are used as a single set.
            labels: Optional labels for each parameter set. When omitted, labels are
                generated from `self.parameters`.
            plot_incidence: Whether to plot the incidence curve.
            plot_prevalence: Whether to plot the prevalence curve.
            figsize: Optional figure size override.
            title: Optional figure title.
            **kwargs: Parameters forwarded to the curve methods when plotting a single
                parameter set.

        Returns:
            The Matplotlib figure.

        Raises:
            ValueError: If neither incidence nor prevalence is selected.
            ValueError: If both `parameter_sets` and `kwargs` are provided.
            ValueError: If `labels` do not match `parameter_sets`.
            ValueError: If `t` is not one-dimensional.

        """
        if not plot_incidence and not plot_prevalence:
            msg = (
                "At least one of `plot_incidence` or `plot_prevalence` must be `True`."
            )
            raise ValueError(msg)

        t_values = _numerical_array_like_to_1d_jax_array(t)
        curve_parameter_sets = _normalize_parameter_sets(
            parameters=self.parameters,
            parameter_sets=parameter_sets,
            kwargs=kwargs,
        )
        incidence_series = (
            [
                jnp.asarray(self.incidence(t_values, **params))
                for params in curve_parameter_sets
            ]
            if plot_incidence
            else None
        )
        prevalence_series = (
            [
                jnp.asarray(self.prevalence(t_values, **params))
                for params in curve_parameter_sets
            ]
            if plot_prevalence
            else None
        )
        return _plot_curve_figure(
            t=t_values,
            parameter_sets=curve_parameter_sets,
            labels=labels,
            incidence_series=incidence_series,
            prevalence_series=prevalence_series,
            parameter_names=self.parameters,
            title=title,
            figsize=figsize,
        )


class LogisticCurve(Curve):
    r"""
    Logistic uptake curve implementation.

    The logistic curve is defined by the following prevalence function:

    This class implements a logistic curve with parameters $m$, $r$, and $s$ which is
    given by:

    $$ f(t \mid m, r, s) = \operatorname{expit}(m)\operatorname{expit}\left(e^r(t-s)\right) $$

    Examples:
        >>> import jax.numpy as jnp
        >>> from vaxflux import LogisticCurve
        >>> t = jnp.array([0.0, 1.0, 2.0, 3.0])
        >>> curve = LogisticCurve()
        >>> prevalence = curve.prevalence(t, m=0.0, r=1.0, s=1.0)
        >>> prevalence
        Array([0.03095159, 0.25      , 0.4690484 , 0.4978322 ], dtype=float32)
        >>> incidence = curve.incidence(t, m=0.0, r=1.0, s=1.0)
        >>> incidence
        Array([0.0789269 , 0.33978522, 0.07892691, 0.00586712], dtype=float32)
    """  # noqa: E501

    def prevalence(  # type: ignore[override]
        self,
        t: NumericalArrayLike,
        m: NumericalArrayLike,
        r: NumericalArrayLike,
        s: NumericalArrayLike,
    ) -> jax.Array:
        """
        Compute the logistic prevalence at time `t`.

        Args:
            t: The time steps to evaluate the prevalence curve at.
            m: The curve's maximum value.
            r: The steepness of the curve.
            s: The x-value of the sigmoid's midpoint.

        Returns:
            The logistic prevalence curve evaluated at time `t`.
        """
        return jsp.expit(m) * jsp.expit(jnp.exp(r) * (t - s))

    def _repr_latex_(self) -> str:
        r"""
        Return a LaTeX representation of the prevalence equation.

        Returns:
            The logistic prevalence equation formatted for Jupyter display.
        """
        return (
            r"$$f(t \mid m, r, s) = "
            r"\operatorname{expit}(m)\operatorname{expit}\left(e^r(t-s)\right)$$"
        )
