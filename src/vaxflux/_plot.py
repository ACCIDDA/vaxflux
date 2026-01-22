from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

try:
    from matplotlib import pyplot as plt

    _matplotlib_available = True
except ImportError:  # pragma: no cover - optional dependency
    _matplotlib_available = False
    plt = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
else:
    Axes = Any  # type: ignore[misc,assignment]
    Figure = Any  # type: ignore[misc,assignment]


def _check_matplotlib_available() -> None:
    """Raise if Matplotlib is not installed."""
    if not _matplotlib_available:
        msg = "matplotlib is required for predictive plotting."
        raise ImportError(msg)


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
    if not _matplotlib_available:  # pragma: no cover - guarded in caller
        msg = "matplotlib is required for predictive plotting."
        raise ImportError(msg)
    if figsize is None:
        figsize = (12, 3 + 3 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=False)
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
