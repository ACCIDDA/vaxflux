import math
import warnings
from collections.abc import Sequence
from numbers import Real
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

import jax

from vaxflux._types import NumericalArrayLike

try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
else:
    Axes = Any  # type: ignore[misc,assignment]
    Figure = Any  # type: ignore[misc,assignment]


def _matplotlib_pyplot() -> ModuleType:
    """
    Return Matplotlib's pyplot module, or raise if it is unavailable.

    Returns:
        The `matplotlib.pyplot` module if it is available.

    Raises:
        ImportError: If the `plot` optional dependency is not available.

    Examples:
        >>> import pytest
        >>> from vaxflux import _plot
        >>> monkeypatch = pytest.MonkeyPatch()
        >>> monkeypatch.setattr(_plot, "plt", "fake_pyplot")
        >>> _plot._matplotlib_pyplot()
        'fake_pyplot'
        >>> monkeypatch.undo()
        >>> monkeypatch = pytest.MonkeyPatch()
        >>> monkeypatch.setattr(_plot, "plt", None)
        >>> _plot._matplotlib_pyplot()
        Traceback (most recent call last):
            ...
        ImportError: The `plot` optional dependency is required for plotting. Please install it with `pip install vaxflux[plot]`.
        >>> monkeypatch.undo()
    """  # noqa: E501
    if plt is None:
        msg = (
            "The `plot` optional dependency is required for plotting. "
            "Please install it with `pip install vaxflux[plot]`."
        )
        raise ImportError(msg)
    return plt


def _plot_predictive_panel(  # noqa: PLR0913
    *,
    ax: Axes,
    quantiles_df: "pd.DataFrame",
    obs_dates: "pd.Series | None",
    obs_values: "pd.Series | None",
    color: str,
    title: str,
    ylabel: str,
    y_max: float,
    quantile_pairs: list[tuple[float, float]],
    alphas: list[float],
) -> None:
    """
    Plot a single predictive panel with median line and shaded intervals.

    Args:
        ax: Axis to draw on.
        quantiles_df: DataFrame with quantile columns indexed by date.
        obs_dates: Observation dates for scatter points.
        obs_values: Observation values for scatter points.
        color: Color for lines and fills.
        title: Panel title.
        ylabel: Y-axis label.
        y_max: Max y-value to scale the axis.
        quantile_pairs: Lower/upper quantile pairs to shade.
        alphas: Alpha values for each shaded interval.

    """
    if 0.5 in quantiles_df.columns:
        ax.plot(
            quantiles_df.index,
            quantiles_df[0.5],
            color=color,
            linewidth=2,
        )
    for (low, high), alpha in zip(quantile_pairs, alphas, strict=False):
        if low in quantiles_df.columns and high in quantiles_df.columns:
            ax.fill_between(
                quantiles_df.index,
                quantiles_df[low],
                quantiles_df[high],
                color=color,
                alpha=alpha,
                linewidth=0,
            )
    if obs_dates is not None and obs_values is not None:
        ax.scatter(obs_dates, obs_values, color=color, s=15)
    ax.set_title(title)
    ax.set_ylim(0, y_max * 1.15)
    ax.set_ylabel(ylabel)


def _init_predictive_axes(
    *,
    n_rows: int,
    n_cols: int,
    figsize: tuple[float, float] | None,
) -> tuple[Figure, list[list[Axes]]]:
    """
    Initialize a grid of Matplotlib axes for predictive plots.

    This helper normalizes the return value so callers can treat the axes as a
    2D list, regardless of how many rows or columns are requested.

    Args:
        n_rows: Number of subplot rows.
        n_cols: Number of subplot columns.
        figsize: Optional figure size override.

    Returns:
        A `(Figure, axes)` tuple where `axes` is a list of rows containing axes.

    """
    pyplot = _matplotlib_pyplot()
    if figsize is None:
        figsize = (12, 3 + 3 * n_rows)
    fig, axes = pyplot.subplots(n_rows, n_cols, figsize=figsize, sharex=False)
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]
    return fig, axes


def _calculate_shaded_quantiles_from_intervals(
    intervals: Sequence[float], min_alpha: float = 0.1, max_alpha: float = 0.25
) -> tuple[list[tuple[float, float]], list[float]]:
    """
    Calculate shaded quantiles from given intervals.

    Args:
        intervals: A sequence of intervals between 0 and 1.
        min_alpha: The minimum alpha value for shading.
        max_alpha: The maximum alpha value for shading.

    Returns:
        A tuple of two lists:

        - A list of tuples representing the lower and upper quantiles for shading.
        - A list of alpha values for shading between the quantiles.

    Raises:
        ValueError: If `intervals` is empty.
        ValueError: If `intervals` are not unique and strictly increasing.
        ValueError: If any value in `intervals` is not between 0 and 1.

    Examples:
        >>> from vaxflux._plot import _calculate_shaded_quantiles_from_intervals
        >>> _calculate_shaded_quantiles_from_intervals([])
        Traceback (most recent call last):
            ...
        ValueError: intervals must contain at least one value.
        >>> _calculate_shaded_quantiles_from_intervals([0.5, 0.8, 0.5])
        Traceback (most recent call last):
            ...
        ValueError: intervals must be unique and strictly increasing.
        >>> _calculate_shaded_quantiles_from_intervals([0.5, 1.2])
        Traceback (most recent call last):
            ...
        ValueError: intervals must be between 0 and 1.

    """
    if not intervals:
        msg = "intervals must contain at least one value."
        raise ValueError(msg)
    intervals = tuple(intervals)
    if sorted(intervals) != list(intervals) or len(set(intervals)) != len(intervals):
        msg = "intervals must be unique and strictly increasing."
        raise ValueError(msg)
    if any(interval <= 0.0 or interval >= 1.0 for interval in intervals):
        msg = "intervals must be between 0 and 1."
        raise ValueError(msg)
    quantiles = []
    for interval in intervals:
        tail = (1.0 - interval) / 2.0
        quantiles.append((tail, 1.0 - tail))
    if len(intervals) == 1:
        alphas = [max_alpha]
    else:
        step = (max_alpha - min_alpha) / (len(intervals) - 1)
        alphas = [max_alpha - step * idx for idx in range(len(intervals))]
    return quantiles, alphas


def _plot_curve_figure(
    *,
    t: jax.Array,
    parameter_sets: Sequence[dict[str, NumericalArrayLike]],
    labels: Sequence[str] | None,
    incidence_series: Sequence[jax.Array] | None,
    prevalence_series: Sequence[jax.Array] | None,
    parameter_names: tuple[str, ...],
    title: str | None,
    figsize: tuple[float, float] | None,
) -> Figure:
    """Plot incidence and/or prevalence curve series with a shared legend."""
    n_cols = int(incidence_series is not None) + int(prevalence_series is not None)
    if figsize is None:
        figsize = (6 * n_cols, 4)
    fig, axes = _init_predictive_axes(n_rows=1, n_cols=n_cols, figsize=figsize)
    axes_list = axes[0]
    curve_labels = _curve_plot_labels(
        parameter_sets=parameter_sets,
        labels=labels,
        parameter_names=parameter_names,
    )
    colors = [f"C{index}" for index in range(len(parameter_sets))]
    incidence_plot_series = (
        [
            (label, y_values, color)
            for label, y_values, color in zip(
                curve_labels, incidence_series, colors, strict=False
            )
        ]
        if incidence_series is not None
        else None
    )
    prevalence_plot_series = (
        [
            (label, y_values, color)
            for label, y_values, color in zip(
                curve_labels, prevalence_series, colors, strict=False
            )
        ]
        if prevalence_series is not None
        else None
    )

    panel_index = 0
    if incidence_plot_series is not None:
        _plot_curve_axis(
            ax=axes_list[panel_index],
            t=t,
            series=incidence_plot_series,
            quantity="incidence",
        )
        panel_index += 1
    if prevalence_plot_series is not None:
        _plot_curve_axis(
            ax=axes_list[panel_index],
            t=t,
            series=prevalence_plot_series,
            quantity="prevalence",
        )

    for ax in axes_list:
        ax.set_xlabel("Time")
    handles, legend_labels = axes_list[0].get_legend_handles_labels()
    fig.legend(
        handles=handles,
        labels=legend_labels,
        title=str(parameter_names),
        loc="center left",
        bbox_to_anchor=(0.9, 0.5),
        prop={"family": "monospace"},
    )
    if title is not None:
        fig.suptitle(title, fontsize=14)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="the figure layout has changed to tight",
            category=UserWarning,
        )
        fig.tight_layout(rect=(0.0, 0.0, 0.9, 1.0))
    return fig


def _plot_curve_axis(
    *,
    ax: Axes,
    t: jax.Array,
    series: Sequence[tuple[str, jax.Array, str]],
    quantity: Literal["incidence", "prevalence"],
) -> None:
    """Draw one curve panel from precomputed series."""
    y_max = 0.0
    for label, y_values, color in series:
        if y_values.size:
            y_max = max(y_max, float(y_values.max()))
        _plot_curve_panel(ax=ax, t=t, y=y_values, color=color, label=label)
    ax.set_title(quantity.capitalize())
    ax.set_ylabel(quantity.capitalize())
    ax.set_ylim(*_curve_plot_ylim(y_max=y_max, quantity=quantity))


def _curve_plot_ylim(
    *,
    y_max: float,
    quantity: Literal["incidence", "prevalence"],
) -> tuple[float, float]:
    """Return y-axis limits for a curve panel."""
    upper = 1.05 * y_max
    if quantity == "prevalence":
        upper = min(1.0, upper)
    return 0.0, upper


def _plot_curve_panel(
    *,
    ax: Axes,
    t: jax.Array,
    y: jax.Array,
    color: str,
    label: str,
) -> None:
    """Draw a single curve line."""
    ax.plot(t, y, color=color, linewidth=2, label=label)


def _curve_plot_labels(
    *,
    parameter_sets: Sequence[dict[str, NumericalArrayLike]],
    labels: Sequence[str] | None,
    parameter_names: tuple[str, ...],
) -> list[str]:
    """
    Return plot labels for the given parameter sets.

    Args:
        parameter_sets: The normalized parameter dictionaries to label.
        labels: Optional user-provided labels to use directly.
        parameter_names: The parameter names that define the display order.

    Returns:
        The formatted legend labels for the parameter sets.

    Raises:
        ValueError: If `labels` is provided but its length does not match
            `parameter_sets`.

    Examples:
        >>> from pprint import pp
        >>> from vaxflux._plot import _curve_plot_labels
        >>> pp(
        ...     _curve_plot_labels(
        ...         parameter_sets=[
        ...             {"m": -1.0, "r": -1.0, "s": 30.0},
        ...             {"m": 0.0, "r": -1.0, "s": 30.0},
        ...             {"m": 1.0, "r": -1.0, "s": 30.0},
        ...         ],
        ...         labels=None,
        ...         parameter_names=("m", "r", "s"),
        ...     ),
        ...     width=22,
        ... )
        ['(-1, -1, 30)',
         '( 0, -1, 30)',
         '( 1, -1, 30)']
        >>> pp(
        ...     _curve_plot_labels(
        ...         parameter_sets=[
        ...             {"m": -0.75, "r": -2.5, "s": 40.0},
        ...             {"m": -0.75, "r": -1.75, "s": 40.0},
        ...             {"m": -0.75, "r": -1.0, "s": 40.0},
        ...         ],
        ...         labels=None,
        ...         parameter_names=("m", "r", "s"),
        ...     ),
        ...     width=22,
        ... )
        ['(-0.75, -2.5 , 40)',
         '(-0.75, -1.75, 40)',
         '(-0.75, -1   , 40)']
        >>> _curve_plot_labels(
        ...     parameter_sets=[{"m": 0.5, "r": 1.0, "s": 1.0}],
        ...     labels=["custom"],
        ...     parameter_names=("m", "r", "s"),
        ... )
        ['custom']
        >>> _curve_plot_labels(
        ...     parameter_sets=[],
        ...     labels=None,
        ...     parameter_names=("m", "r", "s"),
        ... )
        []
        >>> _curve_plot_labels(
        ...     parameter_sets=[{"m": 0.5, "r": 1.0, "s": 1.0}],
        ...     labels=["first", "second"],
        ...     parameter_names=("m", "r", "s"),
        ... )
        Traceback (most recent call last):
            ...
        ValueError: `labels` must have the same length as `parameter_sets`.

    """
    if labels is not None:
        if len(labels) != len(parameter_sets):
            msg = "`labels` must have the same length as `parameter_sets`."
            raise ValueError(msg)
        return list(labels)
    if not parameter_sets:
        return []
    formatted_values = [
        [
            _curve_plot_label_value(params[parameter])
            for parameter in parameter_names
            if parameter in params
        ]
        for params in parameter_sets
    ]
    column_formats = [
        _curve_plot_label_column_format(
            values=[row[column_index] for row in formatted_values]
        )
        for column_index in range(len(formatted_values[0]))
    ]
    return [
        "("
        + ", ".join(
            _curve_plot_align_value(
                value=value,
                integer_width=integer_width,
                fractional_width=fractional_width,
            )
            for column_index, value in enumerate(row)
            for integer_width, fractional_width in [column_formats[column_index]]
        )
        + ")"
        for row in formatted_values
    ]


def _curve_plot_label_value(value: NumericalArrayLike) -> str:
    """
    Return a compact string representation for a parameter value.

    Args:
        value: The parameter value to format.

    Returns:
        A compact string representation suitable for legend labels.

    Examples:
        >>> from vaxflux._plot import _curve_plot_label_value
        >>> _curve_plot_label_value(1.0)
        '1'
        >>> _curve_plot_label_value(-1.75)
        '-1.75'
        >>> _curve_plot_label_value("baseline")
        'baseline'
        >>> _curve_plot_label_value(True)
        'True'

    """
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Real):
        return _curve_plot_real_label(float(value))
    return str(value)


def _curve_plot_real_label(value: float) -> str:
    """
    Return a compact string representation for a real-valued parameter.

    Args:
        value: The real-valued parameter to format.

    Returns:
        A compact decimal representation with unnecessary trailing zeros removed.

    Examples:
        >>> from vaxflux._plot import _curve_plot_real_label
        >>> _curve_plot_real_label(0.0)
        '0'
        >>> _curve_plot_real_label(-0.0)
        '0'
        >>> _curve_plot_real_label(30.0)
        '30'
        >>> _curve_plot_real_label(-1.75)
        '-1.75'
        >>> _curve_plot_real_label(1.234567890123456)
        '1.23456789012346'
        >>> _curve_plot_real_label(float("inf"))
        'inf'
        >>> _curve_plot_real_label(float("nan"))
        'nan'

    """
    if math.isclose(value, 0.0):
        return "0"
    if math.isfinite(value) and value.is_integer():
        return str(int(value))
    return format(value, ".15g")


def _curve_plot_label_column_format(
    *,
    values: Sequence[str],
) -> tuple[int, int]:
    """
    Return integer and fractional widths for one label column.

    Args:
        values: The formatted values in a single label column.

    Returns:
        A pair of `(integer_width, fractional_width)` describing the widest integer
        and fractional parts across the column.

    Examples:
        >>> from vaxflux._plot import _curve_plot_label_column_format
        >>> _curve_plot_label_column_format(values=["-2.5", "-1.75", "-1"])
        (2, 2)
        >>> _curve_plot_label_column_format(values=["20", "30", "40"])
        (2, 0)
        >>> _curve_plot_label_column_format(values=[])
        (0, 0)

    """
    integer_width = 0
    fractional_width = 0
    for value in values:
        integer_part, dot, fractional_part = value.partition(".")
        integer_width = max(integer_width, len(integer_part))
        if dot:
            fractional_width = max(fractional_width, len(fractional_part))
    return integer_width, fractional_width


def _curve_plot_align_value(
    *,
    value: str,
    integer_width: int,
    fractional_width: int,
) -> str:
    """
    Return a decimal-aligned string for one label value.

    Args:
        value: The formatted value to align.
        integer_width: The target width for the integer part.
        fractional_width: The target width for the fractional part.

    Returns:
        The aligned value string.

    Examples:
        >>> from vaxflux._plot import _curve_plot_align_value
        >>> _curve_plot_align_value(
        ...     value="-1.75",
        ...     integer_width=2,
        ...     fractional_width=2,
        ... )
        '-1.75'
        >>> _curve_plot_align_value(
        ...     value="-1",
        ...     integer_width=2,
        ...     fractional_width=2,
        ... )
        '-1   '
        >>> _curve_plot_align_value(
        ...     value="0",
        ...     integer_width=2,
        ...     fractional_width=0,
        ... )
        ' 0'
        >>> _curve_plot_align_value(
        ...     value="20",
        ...     integer_width=2,
        ...     fractional_width=2,
        ... )
        '20   '

    """
    integer_part, dot, fractional_part = value.partition(".")
    aligned = integer_part.rjust(integer_width)
    if fractional_width == 0:
        return aligned
    if dot:
        return f"{aligned}.{fractional_part.ljust(fractional_width)}"
    return f"{aligned}{' ' * (fractional_width + 1)}"
