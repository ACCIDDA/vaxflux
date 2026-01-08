__all__: list[str] = []

import itertools
from itertools import chain
from typing import Any, Literal, Self, cast

import jax.numpy as jnp
import numpyro
import pandas as pd
from jax import Array as JaxArray
from jax.random import key
from numpyro.infer import MCMC, NUTS, Predictive

from vaxflux._covariates import Covariate
from vaxflux._curves import Curve
from vaxflux._interventions import (
    Implementation,
    Intervention,
    _check_interventions_and_implementations,
)
from vaxflux._util import (
    _collect_args,
    _coord_name,
    _validate_and_format_observations,
)
from vaxflux.covariates import CovariateCategories
from vaxflux.dates import (
    DateRange,
    SeasonRange,
    _collect_ranges,
    _infer_ranges_from_observations,
    _validate_ranges,
)


class VaxfluxModel:
    def __init__(self, curve: Curve) -> None:
        """
        Initialize the `VaxfluxModel` with a given uptake curve.

        Args:
            curve: An instance of a `Curve` subclass representing the uptake curve.

        """
        self._curve = curve
        self._seasons: list[SeasonRange] = []
        self._dates: list[DateRange] = []
        self._covariate_categories: list[CovariateCategories] = []
        self._covariates: list[Covariate] = []
        self._observations: pd.DataFrame | None = None
        self._interventions: list[Intervention] = []
        self._implementations: list[Implementation] = []
        self._observation_process_kind: None | Literal["normal"] = None
        self._observation_process_noise: float = 0.05
        self._observation_process_partially_pool_by_season: bool = True
        self._observation_process_partially_pool_by_covariate: bool = False

    def __repr__(self) -> str:
        """
        Return a string representation of the model.

        Returns:
            A string representation of the `VaxfluxModel` instance.
        """
        return f"<vaxflux.VaxfluxModel object at {hex(id(self))}>"

    def add_seasons(self, *args: SeasonRange | list[SeasonRange]) -> Self:
        """
        Add one or more seasons to the model.

        Args:
            *args: One or more `SeasonRange` objects or sequences of `SeasonRange`
                objects.

        Returns:
            The model instance for method chaining.

        Raises:
            TypeError: If any argument is not a `SeasonRange` or a sequence of
                `SeasonRange` objects.

        Examples:
            >>> from vaxflux._curves import LogisticCurve
            >>> from vaxflux.dates import SeasonRange
            >>> model = VaxfluxModel(curve=LogisticCurve())
            >>> model.add_seasons(
            ...     SeasonRange(
            ...         season="2023/2024",
            ...         start_date="2023-12-01",
            ...         end_date="2024-03-31",
            ...     )
            ... )
            <vaxflux.VaxfluxModel object at ...>
            >>> model.add_seasons(
            ...     SeasonRange(
            ...         season="2024/2025",
            ...         start_date="2024-12-01",
            ...         end_date="2025-03-31",
            ...     ),
            ...     SeasonRange(
            ...         season="2025/2026",
            ...         start_date="2025-12-01",
            ...         end_date="2026-03-31",
            ...     ),
            ... )
            <vaxflux.VaxfluxModel object at ...>
            >>> model.add_seasons(
            ...     [
            ...         SeasonRange(
            ...             season="2026/2027",
            ...             start_date="2026-12-01",
            ...             end_date="2027-03-31",
            ...         ),
            ...         SeasonRange(
            ...             season="2027/2028",
            ...             start_date="2027-12-01",
            ...             end_date="2028-03-31",
            ...         ),
            ...     ]
            ... )
            <vaxflux.VaxfluxModel object at ...>
            >>> model.add_seasons("invalid_argument")
            Traceback (most recent call last):
                ...
            TypeError: Arguments must be SeasonRange objects or sequences of SeasonRange objects, got str.

        """  # noqa: E501
        seasons = _collect_ranges(args, SeasonRange, "SeasonRange")
        _validate_ranges(seasons, self._seasons)
        self._seasons.extend(seasons)
        return self

    def add_dates(self, *args: DateRange | list[DateRange]) -> Self:
        """
        Add one or more date ranges to the model.

        Args:
            *args: One or more `DateRange` objects or sequences of `DateRange` objects.

        Returns:
            The model instance for method chaining.

        Raises:
            TypeError: If any argument is not a `DateRange` or a sequence of
                `DateRange` objects.
            ValueError: If duplicate date ranges are found.
            ValueError: If overlapping date ranges are found.

        Examples:
            >>> from vaxflux._curves import LogisticCurve
            >>> from vaxflux.dates import DateRange, SeasonRange
            >>> model = VaxfluxModel(curve=LogisticCurve())
            >>> model.add_seasons(
            ...     SeasonRange(
            ...         season="2023/2024",
            ...         start_date="2023-12-01",
            ...         end_date="2024-03-31",
            ...     )
            ... )
            <vaxflux.VaxfluxModel object at ...>
            >>> model.add_dates(
            ...     DateRange(
            ...         season="2023/2024",
            ...         start_date="2023-12-01",
            ...         end_date="2023-12-07",
            ...         report_date="2023-12-08",
            ...     )
            ... )
            <vaxflux.VaxfluxModel object at ...>
            >>> model.add_dates("invalid_argument")
            Traceback (most recent call last):
                ...
            TypeError: Arguments must be DateRange objects or sequences of DateRange objects, got str.

        """  # noqa: E501
        dates = _collect_ranges(args, DateRange, "DateRange")
        if missing_seasons := {date.season for date in dates} - {
            season.season for season in self._seasons
        }:
            msg = (
                "The following date range seasons have not been added "
                f"to the model: {sorted(missing_seasons)}."
            )
            raise ValueError(msg)
        _validate_ranges(dates, self._dates)
        self._dates.extend(dates)
        return self

    def add_covariate_categories(
        self, *args: CovariateCategories | list[CovariateCategories]
    ) -> Self:
        """
        Add one or more covariate categories to the model.

        Args:
            *args: One or more `CovariateCategories` objects or sequences of
                `CovariateCategories` objects.

        Returns:
            The model instance for method chaining.

        Raises:
            TypeError: If any argument is not a `CovariateCategories` or a sequence of
                `CovariateCategories` objects.

        """
        covariate_categories = _collect_args(
            args, CovariateCategories, "CovariateCategories"
        )
        self._covariate_categories.extend(covariate_categories)
        return self

    def add_covariates(self, *args: Covariate | list[Covariate]) -> Self:
        """
        Add one or more covariates to the model.

        Args:
            *args: One or more `Covariate` objects or sequences of `Covariate` objects.

        Returns:
            The model instance for method chaining.

        Raises:
            TypeError: If any argument is not a `Covariate` or a sequence of
                `Covariate` objects.

        Examples:
            >>> from vaxflux._covariates import PartiallyPooledGaussianCovariate
            >>> from vaxflux._curves import LogisticCurve
            >>> model = VaxfluxModel(curve=LogisticCurve())
            >>> model.add_covariates(
            ...     PartiallyPooledGaussianCovariate(
            ...         parameter="m",
            ...         covariate="age",
            ...         mu=(0.5, 0.1),
            ...         sigma=0.2,
            ...     )
            ... )
            <vaxflux.VaxfluxModel object at ...>
            >>> model.add_covariates(
            ...     [
            ...         PartiallyPooledGaussianCovariate(
            ...             parameter="sigma",
            ...             mu=(0.2, 0.03),
            ...             sigma=0.1,
            ...         ),
            ...     ]
            ... )
            <vaxflux.VaxfluxModel object at ...>
            >>> model.add_covariates("invalid_argument")
            Traceback (most recent call last):
                ...
            TypeError: Arguments must be Covariate objects or sequences of Covariate objects, got str.

        """  # noqa: E501
        covariates = _collect_args(args, Covariate, "Covariate")  # type: ignore[type-abstract]
        self._covariates.extend(covariates)
        return self

    def add_interventions(self, *args: Intervention | list[Intervention]) -> Self:
        """
        Add one or more interventions to the model.

        Args:
            *args: One or more `Intervention` objects or sequences of `Intervention`
                objects.

        Returns:
            The model instance for method chaining.

        Raises:
            TypeError: If any argument is not an `Intervention` or a sequence of
                `Intervention` objects.

        """
        interventions = _collect_args(args, Intervention, "Intervention")
        self._interventions.extend(interventions)
        return self

    def add_implementations(self, *args: Implementation | list[Implementation]) -> Self:
        """
        Add one or more implementations to the model.

        Args:
            *args: One or more `Implementation` objects or sequences of
                `Implementation` objects.

        Returns:
            The model instance for method chaining.

        Raises:
            TypeError: If any argument is not an `Implementation` or a sequence of
                `Implementation` objects.
            ValueError: If an implementation's intervention has not been added to the
                model.

        """
        implementations = _collect_args(args, Implementation, "Implementation")
        if missing_interventions := {
            implementation.intervention for implementation in implementations
        } - {intervention.name for intervention in self._interventions}:
            msg = (
                "The following implementation interventions have not been added "
                f"to the model: {sorted(missing_interventions)}."
            )
            raise ValueError(msg)
        self._implementations.extend(implementations)
        return self

    def add_observations(self, observations: pd.DataFrame) -> Self:
        """
        Add observations to the model.

        Args:
            observations: The observations to add to the model.

        Returns:
            The model instance for method chaining.

        Raises:
            ValueError: If observations were already added to the model.

        """
        if self._observations is not None:
            msg = "Observations have already been added to the model."
            raise ValueError(msg)
        self._observations = _validate_and_format_observations(observations)
        return self

    def add_observation_process(
        self,
        kind: Literal["normal"],
        noise: float,
        *,
        partially_pool_by_season: bool = True,
        partially_pool_by_covariate: bool = False,
    ) -> Self:
        """
        Add an observation process to the model based on the added observations.

        Args:
            kind: The kind of observation process to add.
            noise: The observation noise parameter.
            partially_pool_by_season: Whether to partially pool observation noise by
                season.
            partially_pool_by_covariate: Whether to partially pool observation noise by
                covariate.

        Returns:
            The model instance for method chaining.

        Raises:
            ValueError: If no observations have been added to the model.

        """
        if self._observations is None:
            msg = "No observations have been added to the model, nothing to observe."
            raise ValueError(msg)
        if kind != "normal":
            msg = f"Unsupported observation process kind: {kind}."
            raise ValueError(msg)
        self._observation_process_kind = kind
        self._observation_process_noise = noise
        self._observation_process_partially_pool_by_season = partially_pool_by_season
        self._observation_process_partially_pool_by_covariate = (
            partially_pool_by_covariate
        )
        return self

    def prior_predictive(
        self,
        samples: int,
        random_seed: int = 1,
        *,
        predictive_args: dict[str, Any] | None = None,
    ) -> dict[str, JaxArray]:
        """
        Sample from the prior predictive distribution.

        Args:
            samples: Number of samples to draw.
            random_seed: Seed for the random number generator.
            predictive_args: Additional arguments for the `Predictive` sampler.

        Returns:
            The prior predictive samples.
        """
        predictive_args = predictive_args or {}
        self._pre_model()
        prior_predictive = Predictive(
            self._model, num_samples=samples, **predictive_args
        )
        rng_key = key(random_seed)
        with numpyro.handlers.seed(numpyro.handlers.trace(self._model), rng_key):
            return cast("dict[str, JaxArray]", prior_predictive(rng_key))

    def sample(
        self,
        warmup: int,
        samples: int,
        random_seed: int = 1,
        *,
        mcmc_args: dict[str, Any],
        nuts_args: dict[str, Any],
    ) -> None:
        """
        Sample from the model using MCMC with NUTS.

        Args:
            warmup: Number of warmup (burn-in) steps.
            samples: Number of samples to draw.
            random_seed: Seed for the random number generator.
            mcmc_args: Additional arguments for the MCMC sampler.
            nuts_args: Additional arguments for the NUTS kernel.

        """
        self._pre_model()
        kernel = NUTS(self._model, **nuts_args)
        mcmc = MCMC(kernel, **mcmc_args)
        rng_key = key(random_seed)
        with numpyro.handlers.seed(numpyro.handlers.trace(self._model), rng_key):
            mcmc.run(rng_key, num_warmup=warmup, num_samples=samples)

    def _pre_model(self) -> None:
        """
        Prepare the model by setting up necessary attributes.

        This method should be called before the model is run as it sets up the
        necessary attributes for the model to run correctly.

        """
        _check_interventions_and_implementations(
            self._interventions,
            self._implementations,
            self._seasons,
            self._covariate_categories,
        )
        if self._observations is not None:
            if self._observation_process_kind is None:
                msg = (
                    "Observation process kind must be specified if observations are "
                    "added to the model."
                )
                raise ValueError(msg)
            self._dates = _infer_ranges_from_observations(
                self._observations,
                self._dates,
                "date",
            )
        self._season_names = [season.season for season in self._seasons]
        self._season_name_indices = {
            name: idx for idx, name in enumerate(self._season_names)
        }
        self._season_map = {season.season: season for season in self._seasons}
        self._season_day_counts = {
            season.season: (season.end_date - season.start_date).days + 1
            for season in self._seasons
        }
        self._season_tokens = {
            season_name: _coord_name(season_name) for season_name in self._season_names
        }
        self._covariate_categories_map = {
            category.covariate: category.categories
            for category in self._covariate_categories
        }
        self._covariate_names = list(self._covariate_categories_map.keys())
        self._covariate_name_indices = {
            name: idx for idx, name in enumerate(self._covariate_names)
        }
        self._covariate_category_indices = {
            name: {category: idx for idx, category in enumerate(categories)}
            for name, categories in self._covariate_categories_map.items()
        }
        self._category_combinations = (
            list(
                itertools.product(
                    *[
                        self._covariate_categories_map[name]
                        for name in self._covariate_names
                    ]
                )
            )
            if self._covariate_names
            else [()]
        )
        self._category_combo_indices = {
            combo: idx for idx, combo in enumerate(self._category_combinations)
        }
        self._covariates_by_parameter = {
            param: [cov for cov in self._covariates if cov.parameter == param]
            for param in self._curve.parameters
        }
        self._interventions_by_name = {
            intervention.name: [
                implementation
                for implementation in self._implementations
                if implementation.intervention == intervention.name
            ]
            for intervention in self._interventions
        }

    def _model(self) -> None:
        """Define the model for inference."""
        covariate_values = self._model_sample_covariates()
        summed_parameters = self._model_summed_parameters(covariate_values)
        daily_summed_parameters = self._model_daily_summed_parameters(summed_parameters)
        daily_summed_parameters = self._model_apply_interventions(
            daily_summed_parameters
        )
        observations = self._model_observations(daily_summed_parameters)
        self._model_observation_process(observations)

    def _model_sample_covariate(self, covariate: Covariate) -> jnp.ndarray:
        """
        Sample from a single covariate.

        Args:
            covariate: The covariate to sample from.

        Returns:
            The sampled covariate values.

        """
        covariate_categories: list[str]
        if covariate.covariate is None:
            # If the covariate does not have a covariate name, it is a seasonal
            # covariate, so we use the season names as the categories.
            covariate_categories = self._season_names
            plate_size = len(covariate_categories)
        else:
            # If the covariate has a covariate name, we look up the categories
            # from the map of covariate categories.
            possible_covariate_categories = self._covariate_categories_map.get(
                covariate.covariate
            )
            if possible_covariate_categories is None:
                msg = (
                    f"Covariate categories for '{covariate.covariate}' not found. "
                    "Ensure matching covariate categories are added to the model."
                )
                raise ValueError(msg)
            covariate_categories = list(possible_covariate_categories)
            plate_size = len(covariate_categories) - 1
        # Sample from the covariate and store the sampled values in a
        # deterministic site, optionally padding for non-seasonal covariates.
        covariate.presample()
        with numpyro.plate(f"covariate_{covariate.prefix}", plate_size):
            sampled_values = covariate.sample()
        return cast(
            "jnp.ndarray",
            numpyro.deterministic(
                f"covariate_values_{covariate.prefix}",
                sampled_values
                if covariate.covariate is None
                else jnp.pad(sampled_values, (1, 0), mode="empty"),
            ),
        )

    def _model_sample_covariates(self) -> dict[str, jnp.ndarray]:
        """
        Sample from all covariates.

        Returns:
            A dictionary mapping covariate prefixes to their sampled values.

        """
        # Loop over the covariates and sample from them
        covariate_values: dict[str, jnp.ndarray] = {}
        for covariate in self._covariates:
            covariate_values[covariate.prefix] = self._model_sample_covariate(covariate)
        return covariate_values

    def _model_summed_parameters(
        self, covariate_values: dict[str, jnp.ndarray]
    ) -> dict[str, jnp.ndarray]:
        """
        Calculate the summed parameters for parameter/season/covariate categories.

        Args:
            covariate_values: A dictionary mapping covariate prefixes to their sampled
                values.

        Returns:
            A dictionary mapping (parameter, season, covariate-category combo) strings
            to their summed covariate effects.

        """
        # Now calculate the sums of the covariate values for all of the categories
        # Keys are (parameter, season, covariate-category combo) and values are the
        # summed covariate effects for that combination.
        summed_parameters: dict[str, jnp.ndarray] = {}
        for param in self._curve.parameters:
            for season in self._season_names:
                for category_combo in self._category_combinations:
                    # Collect covariate components that apply to this parameter,
                    # season, and covariate-category combination.
                    components = []
                    for covariate in self._covariates_by_parameter.get(param, []):
                        values = covariate_values[covariate.prefix]
                        if covariate.covariate is None:
                            # Seasonal covariates are indexed by season.
                            season_index = self._season_name_indices[season] + 1
                            components.append(values[season_index])
                            continue
                        # Non-seasonal covariates are indexed by category.
                        category = category_combo[
                            self._covariate_name_indices[covariate.covariate]
                        ]
                        category_index = self._covariate_category_indices[
                            covariate.covariate
                        ][category]
                        components.append(values[category_index])
                    # Sum the components (or use zero when none apply) for this
                    # parameter/season/category combination.
                    summed_name = _coord_name(
                        param, season, *self._category_combo_with_name(category_combo)
                    )
                    summed_parameters[summed_name] = numpyro.deterministic(
                        summed_name,
                        jnp.sum(jnp.stack(components))
                        if components
                        else jnp.asarray(0.0),
                    )
        return summed_parameters

    def _model_daily_summed_parameters(
        self, summed_parameters: dict[str, jnp.ndarray]
    ) -> dict[str, jnp.ndarray]:
        """
        Calculate daily summed parameters for each date range.

        Args:
            summed_parameters: A dictionary mapping (parameter, season,
                covariate-category combo) strings to their summed covariate effects.

        Returns:
            A dictionary mapping daily parameter names to daily summed values.

        """
        daily_summed_parameters: dict[str, jnp.ndarray] = {}
        for name, value in summed_parameters.items():
            matched_season: str | None = None
            for season_name, season_token in self._season_tokens.items():
                if f"_{season_token}_" in name or name.endswith(f"_{season_token}"):
                    matched_season = season_name
                    break
            if matched_season is None:
                msg = f"Could not determine season for summed parameter '{name}'."
                raise ValueError(msg)
            daily_name = _coord_name(name, "daily")
            daily_summed_parameters[daily_name] = numpyro.deterministic(
                daily_name,
                jnp.repeat(value, self._season_day_counts[matched_season]),
            )
        return daily_summed_parameters

    def _model_apply_interventions(
        self, daily_summed_parameters: dict[str, jnp.ndarray]
    ) -> dict[str, jnp.ndarray]:
        """
        Apply interventions to daily summed parameters.

        Args:
            daily_summed_parameters: A dictionary mapping daily parameter names to
                daily summed values.

        Returns:
            A dictionary mapping daily parameter names to adjusted values.

        """
        if not self._interventions or not self._implementations:
            return daily_summed_parameters
        updated_parameters = dict(daily_summed_parameters)
        for intervention in self._interventions:
            intervention_implementations = self._interventions_by_name.get(
                intervention.name,
                [],
            )
            if not intervention_implementations:
                continue
            plate_name = _coord_name(
                "intervention",
                intervention.name,
                "implementations",
            )
            with numpyro.plate(
                f"{plate_name}_plate", len(intervention_implementations)
            ):
                intervention_values = intervention.sample(plate_name)
            for idx, implementation in enumerate(intervention_implementations):
                if implementation.season not in self._season_map:
                    continue
                season_range = self._season_map[implementation.season]
                start_date = implementation.start_date or season_range.start_date
                end_date = implementation.end_date or season_range.end_date
                start_idx = (start_date - season_range.start_date).days
                end_idx = (end_date - season_range.start_date).days
                n_days = (season_range.end_date - season_range.start_date).days + 1
                mask = (jnp.arange(n_days) >= start_idx) & (
                    jnp.arange(n_days) <= end_idx
                )
                for category_combo in self._category_combinations:
                    if implementation.covariate_categories is not None:
                        combo_map = dict(
                            zip(self._covariate_names, category_combo, strict=False)
                        )
                        if not all(
                            combo_map.get(covariate) == category
                            for covariate, category in implementation.covariate_categories.items()  # noqa: E501
                        ):
                            continue
                    daily_name = _coord_name(
                        _coord_name(
                            intervention.parameter,
                            implementation.season,
                            *self._category_combo_with_name(category_combo),
                        ),
                        "daily",
                    )
                    if daily_name not in updated_parameters:
                        continue
                    updated_parameters[daily_name] = (
                        updated_parameters[daily_name] + intervention_values[idx] * mask
                    )
        return updated_parameters

    def _model_observations(
        self, daily_summed_parameters: dict[str, jnp.ndarray]
    ) -> dict[str, jnp.ndarray]:
        """
        Calculate incidence values for each date range.

        Args:
            daily_summed_parameters: A dictionary mapping daily parameter names to
                daily summed values.

        Returns:
            A dictionary mapping date range names to incidence values.

        """
        if not self._dates:
            return {}
        incidence_by_date_range: dict[str, jnp.ndarray] = {}
        for date_range in self._dates:
            season_range = self._season_map[date_range.season]
            start_idx = (date_range.start_date - season_range.start_date).days
            end_idx = (date_range.end_date - season_range.start_date).days
            date_indices = list(range(start_idx, end_idx + 1))
            for category_combo in self._category_combinations:
                daily_params: dict[str, jnp.ndarray] = {}
                for param in self._curve.parameters:
                    base_name = _coord_name(
                        param,
                        date_range.season,
                        *self._category_combo_with_name(category_combo),
                    )
                    daily_name = _coord_name(base_name, "daily")
                    daily_params[param] = daily_summed_parameters[daily_name]
                daily_incidence = jnp.stack(
                    [
                        self._curve.incidence(
                            jnp.asarray([day_idx], dtype=jnp.float32),
                            **{
                                param: daily_params[param][day_idx]
                                for param in self._curve.parameters
                            },
                        )[0]
                        for day_idx in date_indices
                    ]
                )
                incidence_name = _coord_name(
                    "incidence",
                    date_range.season,
                    *self._category_combo_with_name(category_combo),
                    date_range.start_date.isoformat(),
                    date_range.end_date.isoformat(),
                )
                incidence_by_date_range[incidence_name] = numpyro.deterministic(
                    incidence_name,
                    jnp.sum(daily_incidence),
                )
        return incidence_by_date_range

    def _model_observation_process(self, observations: dict[str, jnp.ndarray]) -> None:
        """
        Fit the observation process to the model observations.

        Args:
            observations: A dictionary mapping observation names to incidence values.

        """
        if self._observations is None:
            return
        sigma = self._model_observation_process_sigma()
        for obs_idx, row in self._observations.iterrows():
            if self._covariate_names:
                category_combo = tuple(
                    str(row[covariate_name]) for covariate_name in self._covariate_names
                )
            else:
                category_combo = ()
            category_name = self._category_combo_with_name(category_combo)
            start_date = pd.to_datetime(row["start_date"]).date()
            end_date = pd.to_datetime(row["end_date"]).date()
            observation_name = _coord_name(
                "incidence",
                row["season"],
                *category_name,
                start_date.isoformat(),
                end_date.isoformat(),
            )
            if observation_name not in observations:
                msg = f"Observation key '{observation_name}' not found in model output."
                raise ValueError(msg)
            num_days = (end_date - start_date).days + 1
            if (
                self._observation_process_partially_pool_by_season
                and self._observation_process_partially_pool_by_covariate
            ):
                sigma_value = sigma[
                    self._season_name_indices[row["season"]],
                    self._category_combo_indices[category_combo],
                ]
            elif self._observation_process_partially_pool_by_season:
                sigma_value = sigma[self._season_name_indices[row["season"]]]
            elif self._observation_process_partially_pool_by_covariate:
                sigma_value = sigma[self._category_combo_indices[category_combo]]
            else:
                sigma_value = sigma
            scaled_sigma = sigma_value * jnp.sqrt(num_days)
            numpyro.sample(
                f"observation_{obs_idx}",
                numpyro.distributions.Normal(
                    observations[observation_name],
                    scaled_sigma,
                ),
                obs=row["value"],
            )

    def _model_observation_process_sigma(self) -> jnp.ndarray:
        """
        Sample the observation noise scale for the observation process.

        Returns:
            The sampled observation noise scale, optionally pooled.

        """
        noise_rate = 1.0 / self._observation_process_noise
        if (
            self._observation_process_partially_pool_by_season
            and self._observation_process_partially_pool_by_covariate
        ):
            # Noise parameter is partially pooled by season and covariate category
            with numpyro.plate_stack(
                "observation_sigma",
                (len(self._season_names), len(self._category_combinations)),
            ):
                return cast(
                    "jnp.ndarray",
                    numpyro.sample(
                        "observation_sigma",
                        numpyro.distributions.Exponential(noise_rate),
                    ),
                )
        if self._observation_process_partially_pool_by_season:
            # Noise parameter is partially pooled by season
            with numpyro.plate("observation_sigma_season", len(self._season_names)):
                return cast(
                    "jnp.ndarray",
                    numpyro.sample(
                        "observation_sigma",
                        numpyro.distributions.Exponential(noise_rate),
                    ),
                )
        if self._observation_process_partially_pool_by_covariate:
            # Noise parameter is partially pooled by covariate category combination
            with numpyro.plate(
                "observation_sigma_covariate",
                len(self._category_combinations),
            ):
                return cast(
                    "jnp.ndarray",
                    numpyro.sample(
                        "observation_sigma",
                        numpyro.distributions.Exponential(noise_rate),
                    ),
                )
        # One completely pooled noise parameter
        return cast(
            "jnp.ndarray",
            numpyro.sample(
                "observation_sigma",
                numpyro.distributions.Exponential(noise_rate),
            ),
        )

    def _category_combo_with_name(
        self, category_combo: tuple[str, ...]
    ) -> tuple[str, ...]:
        """
        Generate a name for a covariate category combination.

        Args:
            category_combo: A tuple of covariate category names.

        Returns:
            A tuple representing the full name of the covariate category combination.

        """
        return tuple(
            chain.from_iterable(
                zip(self._covariate_names, category_combo, strict=False)
            )
        )
