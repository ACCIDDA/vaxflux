import warnings
from collections.abc import Sequence
from functools import cached_property
from itertools import product
from typing import Any, Literal, cast

import arviz as az
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure

try:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
except ImportError:
    Line2D = Any  # type: ignore[misc,assignment]
    Rectangle = Any  # type: ignore[misc,assignment]

from vaxflux._plot import (
    _calculate_shaded_quantiles_from_intervals,
    _init_predictive_axes,
    _plot_predictive_panel,
)
from vaxflux._util import _compute_quantiles_and_max, _coord_name


class VaxfluxInferenceData(az.InferenceData):
    """
    Container for inference data specific to vaxflux.

    This class extends
    [ArviZ's `InferenceData`](https://python.arviz.org/en/stable/api/generated/arviz.InferenceData.html)
    to include specialized functionality for vaxflux models. This allows for users to
    easily use this object with functions that expect ArviZ `InferenceData` while also
    providing vaxflux-specific methods and attributes.

    """

    observations: pd.DataFrame | None = None

    @classmethod
    def from_numpyro(cls, *args: Any, **kwargs: Any) -> "VaxfluxInferenceData":
        """
        Create inference data from NumPyro outputs via ArviZ.

        For more details, see
        [ArviZ's `from_numpyro` function](https://python.arviz.org/en/stable/api/generated/arviz.from_numpyro.html).

        Args:
            *args: Positional arguments forwarded to `arviz.from_numpyro`.
            **kwargs: Keyword arguments forwarded to `arviz.from_numpyro`.

        Returns:
            A `VaxfluxInferenceData` instance populated from NumPyro results.

        """
        observations = cast("pd.DataFrame | None", kwargs.pop("observations", None))
        idata = cast(
            "az.InferenceData",
            az.from_numpyro(*args, **kwargs),  # type: ignore[no-untyped-call]
        )
        idata.__class__ = cls
        idata.observations = observations.copy() if observations is not None else None  # type: ignore[attr-defined]
        return cast("VaxfluxInferenceData", idata)

    @cached_property
    def merged_prior(self) -> xr.Dataset:
        """
        Return a merged prior dataset combining prior and prior predictive.

        Returns:
            An xarray Dataset with prior and prior predictive variables.

        """
        prior = getattr(self, "prior", None)
        prior_predictive = getattr(self, "prior_predictive", None)
        if prior is None and prior_predictive is None:
            msg = "Prior predictive data not available in this inference object."
            raise ValueError(msg)
        datasets = [ds for ds in (prior, prior_predictive) if ds is not None]
        return cast("xr.Dataset", xr.merge(datasets, compat="no_conflicts"))

    @cached_property
    def merged_posterior(self) -> xr.Dataset:
        """
        Return a merged posterior dataset combining posterior and posterior predictive.

        Returns:
            An xarray Dataset with posterior and posterior predictive variables.

        """
        posterior = getattr(self, "posterior", None)
        posterior_predictive = getattr(self, "posterior_predictive", None)
        if posterior is None and posterior_predictive is None:
            msg = "Posterior predictive data not available in this inference object."
            raise ValueError(msg)
        datasets = [ds for ds in (posterior, posterior_predictive) if ds is not None]
        return cast("xr.Dataset", xr.merge(datasets, compat="no_conflicts"))

    @cached_property
    def coords(self) -> dict[str, list[str]]:
        """
        Return model coordinates without chain/draw dimensions.

        Returns:
            The coordinate mapping for the merged prior or posterior dataset without
            chain and draw dimensions (which differ between prior/posterior).

        """
        if (
            getattr(self, "prior", None) is not None
            or getattr(self, "prior_predictive", None) is not None
        ):
            dataset = self.merged_prior
        elif (
            getattr(self, "posterior", None) is not None
            or getattr(self, "posterior_predictive", None) is not None
        ):
            dataset = self.merged_posterior
        else:
            msg = "No data available in this inference object."
            raise ValueError(msg)
        coords = {}
        for name in dataset.coords:
            if name in {"chain", "draw"}:
                continue
            coords[str(name)] = cast(
                "list[str]", dataset.coords[name].to_numpy().tolist()
            )
        return coords

    @cached_property
    def covariate_categories(self) -> dict[str, dict[str, str]]:
        """
        A mapping of covariate names to their category labels.

        Returns:
            A dictionary mapping covariate names to dictionaries of category labels
            where each inner dictionary maps coordinate-safe category names to the
            original category labels.

        """
        covariate_names = {
            key[: -len("_categories")]
            for key in self.coords
            if key.endswith("_categories") and not key.endswith("_categories_short")
        }
        covariate_category_map: dict[str, dict[str, str]] = {}
        for covariate_name in covariate_names:
            coord_key = f"{covariate_name}_categories"
            if coord_key not in self.coords:
                continue
            categories = self.coords[coord_key]
            covariate_category_map[covariate_name] = {
                _coord_name(str(cat)): str(cat) for cat in categories
            }
        return covariate_category_map

    @cached_property
    def prior_observations(self) -> pd.DataFrame:
        """
        Return prior observations as a formatted DataFrame.

        Returns:
            A DataFrame with observation metadata and values. Will contain the columns
            'chain', 'draw', 'season', 'season_start_date', 'season_end_date',
            'start_date', 'end_date', 'report_date', any covariate columns,
            'type', and 'value'.

        """
        return self._observations_from_dataset(self.merged_prior)

    @cached_property
    def posterior_observations(self) -> pd.DataFrame:
        """
        Return posterior observations as a formatted DataFrame.

        Returns:
            A DataFrame with observation metadata and values. Will contain the columns
            'chain', 'draw', 'season', 'season_start_date', 'season_end_date',
            'start_date', 'end_date', 'report_date', any covariate columns,
            'type', and 'value'.

        """
        return self._observations_from_dataset(self.merged_posterior)

    def _observations_from_dataset(self, dataset: xr.Dataset) -> pd.DataFrame:
        """
        Build an observations DataFrame from an ArviZ dataset.

        This extracts incidence curve parameters and reconstructs observation metadata
        from stored coordinates and variable naming conventions.

        """
        observations: list[pd.DataFrame] = []

        # Find incidence variables to decode into observation rows
        incidence_vars = [
            name for name in dataset.data_vars if str(name).startswith("incidence_")
        ]
        if not incidence_vars:
            msg = "No incidence variables found in the inference data."
            raise ValueError(msg)

        covariate_names = self.coords.get("covariate_names", [])
        if not covariate_names:
            covariate_names = sorted(
                key[: -len("_categories")]
                for key in self.coords
                if key.endswith("_categories") and not key.endswith("_categories_short")
            )
        covariate_categories = {
            name: self.coords.get(f"{name}_categories", []) for name in covariate_names
        }
        season_labels = self.coords.get("season", [])
        date_dims = [dim for dim in dataset.dims if str(dim).startswith("date_ranges_")]

        # Convert each season/category combination into observation rows
        for date_dim in date_dims:
            season_token = str(date_dim)[len("date_ranges_") :]
            season = next(
                (
                    season_name
                    for season_name in season_labels
                    if _coord_name(season_name) == season_token
                ),
                season_token,
            )
            category_combos = (
                list(product(*(covariate_categories[name] for name in covariate_names)))
                if covariate_names
                else [()]
            )
            for combo in category_combos:
                name_parts = ["incidence", season]
                for covariate_name, category_value in zip(
                    covariate_names, combo, strict=False
                ):
                    name_parts.extend([covariate_name, category_value])
                var_name = _coord_name(*name_parts)
                if var_name not in dataset.data_vars:
                    continue
                data = dataset[var_name]
                df = data.to_dataframe(name="value").reset_index()

                # Parse start/end/report dates from the date-range coordinate
                date_values = df[date_dim].str.split("_", n=1, expand=True)
                df["start_date"] = pd.to_datetime(date_values[0])
                df["end_date"] = pd.to_datetime(date_values[1])
                df["report_date"] = df["end_date"]
                df["season"] = season

                # Fill season start/end dates from the season day coordinates if present
                days_key = _coord_name("days", season)
                if days_key in dataset.coords:
                    days = dataset.coords[days_key].to_numpy().tolist()
                    df["season_start_date"] = (
                        pd.to_datetime(days[0]) if days else pd.NaT
                    )
                    df["season_end_date"] = pd.to_datetime(days[-1]) if days else pd.NaT
                else:
                    df["season_start_date"] = pd.NaT
                    df["season_end_date"] = pd.NaT

                for covariate_name, category_value in zip(
                    covariate_names, combo, strict=False
                ):
                    df[covariate_name] = category_value

                # Finalize this block of observations and move to the next
                df["type"] = "incidence"
                df = df.drop(columns=[date_dim])
                observations.append(df)

        if not observations:
            msg = "No observation data could be constructed from incidence outputs."
            raise ValueError(msg)

        df = pd.concat(observations, ignore_index=True)
        return df[
            [
                "chain",
                "draw",
                "season",
                "season_start_date",
                "season_end_date",
                "start_date",
                "end_date",
                "report_date",
                *covariate_names,
                "type",
                "value",
            ]
        ]

    def prior_prevalence_scenarios(
        self,
        scenarios: dict[str, tuple[float, float]],
    ) -> pd.DataFrame:
        """
        Summarize prior end-of-season prevalence scenarios.

        Args:
            scenarios: Mapping of scenario name to (low, high) quantile bounds.

        Returns:
            A DataFrame with scenario quantiles and prevalence bounds.

        """
        return self._prevalence_scenarios_from_observations(
            observations=self.prior_observations,
            scenarios=scenarios,
        )

    def posterior_prevalence_scenarios(
        self,
        scenarios: dict[str, tuple[float, float]],
    ) -> pd.DataFrame:
        """
        Summarize posterior end-of-season prevalence scenarios.

        Args:
            scenarios: Mapping of scenario name to (low, high) quantile bounds.

        Returns:
            A DataFrame with scenario quantiles and prevalence bounds.

        """
        return self._prevalence_scenarios_from_observations(
            observations=self.posterior_observations,
            scenarios=scenarios,
        )

    def _prevalence_scenarios_from_observations(
        self,
        *,
        observations: pd.DataFrame,
        scenarios: dict[str, tuple[float, float]],
    ) -> pd.DataFrame:
        """
        Build prevalence scenario summaries from an observations DataFrame.

        Args:
            observations: Observations data with incidence values.
            scenarios: Mapping of scenario name to (low, high) quantile bounds.

        Returns:
            A DataFrame with scenario quantiles and prevalence bounds.

        Raises:
            ValueError: If scenarios are empty or contain invalid quantiles.
            ValueError: If any of the scenario quantiles are out of the range [0, 1].
            ValueError: If any of the scenario quantiles do not satisfy low < high.

        """
        if not scenarios:
            msg = "scenarios must not be empty."
            raise ValueError(msg)

        standard_cols = {
            "chain",
            "draw",
            "season",
            "season_start_date",
            "season_end_date",
            "start_date",
            "end_date",
            "report_date",
            "type",
            "value",
        }
        covariate_cols = [
            col for col in observations.columns if col not in standard_cols
        ]
        group_cols = ["chain", "draw", "season", *covariate_cols]
        end_prevalence = observations.groupby(group_cols)["value"].sum().reset_index()
        summary_group_cols = ["season", *covariate_cols]

        invalid_range = [
            scenario
            for scenario, (low_q, high_q) in scenarios.items()
            if not (0.0 <= low_q <= 1.0 and 0.0 <= high_q <= 1.0)
        ]
        if invalid_range:
            msg = (
                "Scenario quantiles must be within [0, 1]. "
                f"Invalid scenarios: {', '.join(sorted(invalid_range))}."
            )
            raise ValueError(msg)
        invalid_order = [
            scenario
            for scenario, (low_q, high_q) in scenarios.items()
            if high_q <= low_q
        ]
        if invalid_order:
            msg = (
                "Scenario quantiles must satisfy low < high. "
                f"Invalid scenarios: {', '.join(sorted(invalid_order))}."
            )
            raise ValueError(msg)

        rows: list[dict[str, str | float]] = []
        for scenario, (low_q, high_q) in scenarios.items():
            for group_keys, group_data in end_prevalence.groupby(summary_group_cols):
                key_tuple = (
                    group_keys if isinstance(group_keys, tuple) else (group_keys,)
                )
                row: dict[str, float | str] = {
                    "scenario": scenario,
                    "low_quantile": low_q,
                    "high_quantile": high_q,
                    "low_prevalence": float(group_data["value"].quantile(low_q)),
                    "high_prevalence": float(group_data["value"].quantile(high_q)),
                }
                row.update(
                    cast(
                        "dict[str, float | str]",
                        dict(zip(summary_group_cols, key_tuple, strict=False)),
                    )
                )
                rows.append(row)

        df = pd.DataFrame(rows)
        ordered_cols = [
            "season",
            *covariate_cols,
            "scenario",
            "low_quantile",
            "high_quantile",
            "low_prevalence",
            "high_prevalence",
        ]
        return df[ordered_cols]

    def plot_prior_predictive(
        self,
        *,
        intervals: Sequence[float] = (0.5, 0.8, 0.95),
        plot_incidence: bool = True,
        plot_prevalence: bool = True,
        plot_observations: bool | None = None,
        predictive: Literal["latent", "observation"] = "latent",
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        """
        Plot prior predictive checks with shaded intervals.

        Args:
            intervals: Central predictive intervals to plot.
            plot_incidence: Whether to plot incidence panels.
            plot_prevalence: Whether to plot prevalence panels.
            plot_observations: Whether to plot observed data points. When `None`,
                uses `self.observations` availability.
            predictive: Whether to plot latent curve intervals or observation-level
                predictive intervals.
            figsize: Optional figure size override.

        Returns:
            The Matplotlib figure.

        """
        predictive_label = (
            "Observation Predictive" if predictive == "observation" else "Predictive"
        )
        return self._plot_predictive(
            data=self._predictive_observations(
                predictive=predictive,
                fallback=self.prior_observations,
                dataset=(
                    self.merged_prior
                    if predictive == "observation"
                    else getattr(self, "prior_predictive", None)
                ),
            ),
            title=f"Prior {predictive_label} Checks",
            intervals=intervals,
            plot_incidence=plot_incidence,
            plot_prevalence=plot_prevalence,
            plot_observations=plot_observations,
            figsize=figsize,
        )

    def plot_posterior_predictive(
        self,
        *,
        intervals: Sequence[float] = (0.5, 0.8, 0.95),
        plot_incidence: bool = True,
        plot_prevalence: bool = True,
        plot_observations: bool | None = None,
        predictive: Literal["latent", "observation"] = "latent",
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        """
        Plot posterior predictive checks with shaded intervals.

        Args:
            intervals: Central predictive intervals to plot.
            plot_incidence: Whether to plot incidence panels.
            plot_prevalence: Whether to plot prevalence panels.
            plot_observations: Whether to plot observed data points. When `None`,
                uses `self.observations` availability.
            predictive: Whether to plot latent curve intervals or observation-level
                predictive intervals.
            figsize: Optional figure size override.

        Returns:
            The Matplotlib figure.

        """
        predictive_label = (
            "Observation Predictive" if predictive == "observation" else "Predictive"
        )
        return self._plot_predictive(
            data=self._predictive_observations(
                predictive=predictive,
                fallback=self.posterior_observations,
                dataset=(
                    self.merged_posterior
                    if predictive == "observation"
                    else getattr(self, "posterior_predictive", None)
                ),
            ),
            title=f"Posterior {predictive_label} Checks",
            intervals=intervals,
            plot_incidence=plot_incidence,
            plot_prevalence=plot_prevalence,
            plot_observations=plot_observations,
            figsize=figsize,
        )

    def _plot_predictive(
        self,
        *,
        data: pd.DataFrame,
        title: str,
        intervals: Sequence[float],
        plot_incidence: bool,
        plot_prevalence: bool,
        plot_observations: bool | None,
        figsize: tuple[float, float] | None,
    ) -> Figure:
        """
        Plot predictive checks with shaded intervals.

        Args:
            data: The prior/posterior observations DataFrame to plot.
            title: The plot title.
            intervals: Central predictive intervals to plot.
            plot_incidence: Whether to plot incidence panels.
            plot_prevalence: Whether to plot prevalence panels.
            plot_observations: Whether to plot observed data points. When `None`,
                uses `self.observations` availability.
            figsize: Optional figure size override.

        Returns:
            The Matplotlib figure.

        Raises:
            ValueError: If neither incidence nor prevalence is selected for plotting.
            ValueError: If `plot_observations` is `True` but no observations are set.

        """
        # Input validation
        if not plot_incidence and not plot_prevalence:
            msg = (
                "At least one of `plot_incidence` or `plot_prevalence` must be `True`."
            )
            raise ValueError(msg)

        plot_data, seasons, categories, colors = self._prepare_predictive_plot_data(
            data
        )
        plot_observations, obs_plot_data = self._resolve_plot_observations(
            plot_observations=plot_observations
        )

        fig, axes = _init_predictive_axes(
            n_rows=len(seasons),
            n_cols=(1 if plot_incidence else 0) + (1 if plot_prevalence else 0),
            figsize=figsize,
        )

        quantiles, alphas = _calculate_shaded_quantiles_from_intervals(intervals)

        quantile_levels = [q for pair in quantiles for q in pair] + [0.5]

        quantile_map, max_inc, max_prev = self._prepare_predictive_quantiles(
            plot_data=plot_data,
            obs_plot_data=obs_plot_data,
            seasons=seasons,
            categories=categories,
            quantile_levels=quantile_levels,
            plot_incidence=plot_incidence,
            plot_prevalence=plot_prevalence,
        )

        for row_idx, season in enumerate(seasons):
            row_axes = axes[row_idx]
            self._plot_predictive_season_row(
                row_axes=row_axes,
                season=season,
                categories=categories,
                colors=colors,
                quantile_map=quantile_map,
                quantile_pairs=quantiles,
                alphas=alphas,
                plot_incidence=plot_incidence,
                plot_prevalence=plot_prevalence,
            )

            for label in [
                *row_axes[0].get_xticklabels(),
                *row_axes[-1].get_xticklabels(),
            ]:
                label.set_rotation(30)

        self._apply_shared_ylim(
            axes=axes,
            max_inc=max_inc,
            max_prev=max_prev,
            plot_incidence=plot_incidence,
            plot_prevalence=plot_prevalence,
        )

        legend_handles: list[Any] = [
            Line2D([0], [0], color=colors[cat], lw=2, label=cat) for cat in categories
        ]
        legend_handles.append(
            Line2D([0], [0], color="black", lw=2, linestyle="-", label="Median")
        )
        for interval, alpha in zip(intervals, alphas, strict=False):
            legend_handles.append(
                Rectangle(
                    (0, 0),
                    1,
                    1,
                    color="black",
                    alpha=alpha,
                    label=f"{int(interval * 100)}% interval",
                )
            )
        fig.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.0, 0.5))
        fig.suptitle(title, fontsize=14)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the figure layout has changed to tight",
                category=UserWarning,
            )
            fig.tight_layout()
        return fig

    def _predictive_observations(
        self,
        *,
        predictive: Literal["latent", "observation"],
        fallback: pd.DataFrame,
        dataset: xr.Dataset | None,
    ) -> pd.DataFrame:
        """
        Resolve predictive observation data for plotting.

        Args:
            predictive: Whether to return latent curve or observation-level draws.
            fallback: Latent observation DataFrame used when predictive is "latent".
            dataset: Dataset containing observation-level samples.

        Returns:
            A DataFrame of observations for plotting.

        Raises:
            ValueError: If predictive samples are unavailable or invalid.

        """
        if predictive == "latent":
            return fallback
        if dataset is None:
            msg = (
                "Observation-level predictive samples are not available; "
                "ensure prior or posterior predictive data is present."
            )
            raise ValueError(msg)
        return self._observations_from_observation_samples(dataset)

    def _observations_from_observation_samples(
        self,
        dataset: xr.Dataset,
    ) -> pd.DataFrame:
        """
        Build an observations DataFrame from observation-level samples.

        Args:
            dataset: Dataset containing `observation_sim` samples and coordinates.

        Returns:
            A DataFrame with observation metadata and values.

        Raises:
            ValueError: If observation samples or labels are missing.

        """
        if "observation_sim" not in dataset.data_vars:
            msg = "No observation samples found in the inference data."
            raise ValueError(msg)
        if "observation" not in dataset.coords:
            msg = "No observation labels found in the inference data."
            raise ValueError(msg)

        covariate_names = self.coords.get("covariate_names", [])
        if not covariate_names:
            covariate_names = sorted(
                key[: -len("_categories")]
                for key in self.coords
                if key.endswith("_categories") and not key.endswith("_categories_short")
            )
        label_map = self._parse_observation_labels(
            labels=dataset.coords["observation"].to_numpy().tolist(),
            covariate_names=list(covariate_names),
        )
        data = dataset["observation_sim"].to_dataframe(name="value").reset_index()
        meta = pd.DataFrame(label_map)
        df = data.merge(meta, on="observation", how="left")
        if df[["season", "start_date", "end_date"]].isna().any().any():
            msg = "Unable to map observation samples to observation metadata."
            raise ValueError(msg)

        for col in ("season_start_date", "season_end_date", "report_date"):
            if col not in df.columns:
                df[col] = pd.NaT
        df["type"] = "incidence"
        return df[
            [
                "chain",
                "draw",
                "season",
                "season_start_date",
                "season_end_date",
                "start_date",
                "end_date",
                "report_date",
                *covariate_names,
                "type",
                "value",
            ]
        ]

    def _parse_observation_labels(
        self,
        *,
        labels: list[str],
        covariate_names: list[str],
    ) -> list[dict[str, Any]]:
        """
        Parse observation labels into structured metadata.

        Args:
            labels: Observation labels created by the model.
            covariate_names: Covariate names in the expected label order.

        Returns:
            A list of metadata dictionaries keyed by "observation".

        Raises:
            ValueError: If a label cannot be parsed.

        """
        metadata: list[dict[str, Any]] = []
        for label in labels:
            parts = str(label).split("|")
            if len(parts) < 3:
                msg = f"Invalid observation label: {label}."
                raise ValueError(msg)
            season, start_date, end_date = parts[:3]
            covariate_values: dict[str, str] = {}
            for item in parts[3:]:
                if "=" not in item:
                    msg = f"Invalid observation label: {label}."
                    raise ValueError(msg)
                key, value = item.split("=", 1)
                covariate_values[key] = value
            for covariate_name in covariate_names:
                covariate_values.setdefault(covariate_name, "")
            metadata.append(
                {
                    "observation": label,
                    "season": season,
                    "start_date": pd.to_datetime(start_date),
                    "end_date": pd.to_datetime(end_date),
                    "report_date": pd.to_datetime(end_date),
                    **covariate_values,
                }
            )
        return metadata

    def _resolve_plot_observations(
        self,
        *,
        plot_observations: bool | None,
    ) -> tuple[bool, pd.DataFrame | None]:
        """
        Resolve observation plotting flags and data.

        Args:
            plot_observations: Whether to plot observations, or `None` to infer from
                availability.

        Returns:
            A tuple of the resolved plotting flag and the prepared observations data.

        Raises:
            ValueError: If plotting is requested but no observations are available.

        """
        plot_observations = (
            (self.observations is not None)
            if plot_observations is None
            else plot_observations
        )
        if plot_observations and self.observations is None:
            msg = "plot_observations=True requires observations to be set."
            raise ValueError(msg)
        if plot_observations and self.observations is not None:
            obs_plot_data, _, _, _ = self._prepare_predictive_plot_data(
                self.observations
            )
            return plot_observations, obs_plot_data
        return plot_observations, None

    def _prepare_predictive_quantiles(
        self,
        *,
        plot_data: pd.DataFrame,
        obs_plot_data: pd.DataFrame | None,
        seasons: Sequence[str],
        categories: Sequence[str],
        quantile_levels: Sequence[float],
        plot_incidence: bool,
        plot_prevalence: bool,
    ) -> tuple[dict[tuple[str, str], dict[str, pd.DataFrame | None]], float, float]:
        """
        Compute predictive quantiles and maxima for plotting.

        Args:
            plot_data: Prepared predictive data for plotting.
            obs_plot_data: Prepared observation data for plotting, if available.
            seasons: Seasons present in the data.
            categories: Category labels to plot.
            quantile_levels: Quantiles to compute for intervals and median.
            plot_incidence: Whether incidence quantiles are needed.
            plot_prevalence: Whether prevalence quantiles are needed.

        Returns:
            A mapping of season/category to quantiles and observations plus maxima for
            incidence and prevalence.

        """
        quantile_map: dict[tuple[str, str], dict[str, pd.DataFrame | None]] = {}
        max_inc = 0.0
        max_prev = 0.0

        for season in seasons:
            season_obs = plot_data[plot_data["season"] == season]
            for category in categories:
                cat_obs = season_obs[season_obs["category_label"] == category]
                obs_cat = None
                if obs_plot_data is not None:
                    obs_cat = obs_plot_data[
                        (obs_plot_data["season"] == season)
                        & (obs_plot_data["category_label"] == category)
                    ]
                value_quantiles = pd.DataFrame()
                prev_quantiles = pd.DataFrame()
                if plot_incidence:
                    value_quantiles, max_inc = _compute_quantiles_and_max(
                        cat_obs,
                        "value",
                        quantile_levels,
                        max_inc,
                    )
                if plot_prevalence:
                    prev_quantiles, max_prev = _compute_quantiles_and_max(
                        cat_obs,
                        "prevalence",
                        quantile_levels,
                        max_prev,
                    )
                quantile_map[(season, category)] = {
                    "value": value_quantiles,
                    "prevalence": prev_quantiles,
                    "obs": obs_cat,
                }
        return quantile_map, max_inc, max_prev

    def _plot_predictive_season_row(  # noqa: PLR0913
        self,
        *,
        row_axes: list[Any],
        season: str,
        categories: Sequence[str],
        colors: dict[str, str],
        quantile_map: dict[tuple[str, str], dict[str, pd.DataFrame | None]],
        quantile_pairs: list[tuple[float, float]],
        alphas: list[float],
        plot_incidence: bool,
        plot_prevalence: bool,
    ) -> None:
        """
        Plot predictive panels for a single season row.

        Args:
            row_axes: Axes for the season row.
            season: Season label for titles.
            categories: Category labels to plot.
            colors: Category color mapping.
            quantile_map: Precomputed quantiles and observations by season/category.
            quantile_pairs: Lower/upper quantile pairs for shaded intervals.
            alphas: Alpha values for shaded intervals.
            plot_incidence: Whether to plot incidence panels.
            plot_prevalence: Whether to plot prevalence panels.

        """
        for category in categories:
            entry = quantile_map[(season, category)]
            obs_cat = entry["obs"]
            col_idx = 0
            if plot_incidence:
                ax_inc = row_axes[col_idx]
                col_idx += 1
                _plot_predictive_panel(
                    ax=ax_inc,
                    quantiles_df=(
                        entry["value"] if entry["value"] is not None else pd.DataFrame()
                    ),
                    obs_dates=obs_cat["mid_date"] if obs_cat is not None else None,
                    obs_values=obs_cat["value"] if obs_cat is not None else None,
                    color=colors.get(category, "C0"),
                    title=f"{season} incidence",
                    ylabel="Incidence",
                    y_max=1.0,
                    quantile_pairs=quantile_pairs,
                    alphas=alphas,
                )

            if plot_prevalence:
                ax_prev = row_axes[col_idx]
                _plot_predictive_panel(
                    ax=ax_prev,
                    quantiles_df=(
                        entry["prevalence"]
                        if entry["prevalence"] is not None
                        else pd.DataFrame()
                    ),
                    obs_dates=obs_cat["mid_date"] if obs_cat is not None else None,
                    obs_values=obs_cat["prevalence"] if obs_cat is not None else None,
                    color=colors.get(category, "C0"),
                    title=f"{season} prevalence",
                    ylabel="Prevalence",
                    y_max=1.0,
                    quantile_pairs=quantile_pairs,
                    alphas=alphas,
                )

    def _apply_shared_ylim(
        self,
        *,
        axes: list[list[Any]],
        max_inc: float,
        max_prev: float,
        plot_incidence: bool,
        plot_prevalence: bool,
    ) -> None:
        """
        Apply shared y-axis limits across panels.

        Args:
            axes: Axes grid for the figure.
            max_inc: Maximum incidence value across quantiles.
            max_prev: Maximum prevalence value across quantiles.
            plot_incidence: Whether incidence panels are present.
            plot_prevalence: Whether prevalence panels are present.

        """
        if plot_incidence:
            inc_ylim = max_inc * 1.15 if max_inc > 0 else 1.0
            for row_axes in axes:
                row_axes[0].set_ylim(0, inc_ylim)
        if plot_prevalence:
            prev_ylim = max_prev * 1.15 if max_prev > 0 else 1.0
            for row_axes in axes:
                row_axes[-1].set_ylim(0, prev_ylim)

    def _prepare_predictive_plot_data(
        self,
        data: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str], list[str], dict[str, str]]:
        plot_data = data.copy()
        plot_data["start_date"] = pd.to_datetime(plot_data["start_date"])
        plot_data["end_date"] = pd.to_datetime(plot_data["end_date"])
        plot_data["mid_date"] = (
            plot_data["start_date"]
            + (plot_data["end_date"] - plot_data["start_date"]) / 2
        )
        plot_data = plot_data.sort_values("mid_date")
        group_cols = [
            col for col in ["chain", "draw", "season"] if col in plot_data.columns
        ]
        covariate_names = sorted(self.covariate_categories)
        if covariate_names:
            group_cols.extend(covariate_names)
            label_cols = covariate_names
        else:
            label_cols = []
        plot_data["prevalence"] = plot_data.groupby(group_cols)["value"].cumsum()
        plot_data["category_label"] = (
            plot_data[label_cols].astype(str).agg("|".join, axis=1)
            if label_cols
            else "all"
        )
        seasons = sorted(plot_data["season"].unique())
        categories = sorted(plot_data["category_label"].unique())
        colors = {cat: f"C{idx}" for idx, cat in enumerate(categories)}
        return plot_data, seasons, categories, colors
