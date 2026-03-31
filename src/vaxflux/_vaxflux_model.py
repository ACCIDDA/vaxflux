__all__: list[str] = []

import itertools
import math
from collections import Counter
from datetime import timedelta
from itertools import chain
from typing import Any, Literal, Self, cast

import jax.numpy as jnp
import numpyro
import pandas as pd
from jax.random import key, split
from numpyro.infer import MCMC, NUTS, Predictive

from vaxflux._covariate_categories import CovariateCategories
from vaxflux._covariate_index_info import CovariateIndexInfo
from vaxflux._covariates import Covariate
from vaxflux._curves import Curve
from vaxflux._dates import (
    DateRange,
    SeasonRange,
    _collect_ranges,
    _infer_ranges_from_observations,
    _validate_ranges,
)
from vaxflux._interventions import (
    Implementation,
    Intervention,
    _check_interventions_and_implementations,
)
from vaxflux._util import (
    _collect_args,
    _coord_name,
)
from vaxflux._vaxflux_inference_data import VaxfluxInferenceData
from vaxflux._vaxflux_observations import VaxfluxObservations

try:
    from numpyro import render_model as _render_model
except ImportError:
    # Fallback for older NumPyro
    from numpyro.contrib.render import render_model as _render_model


class VaxfluxModel:
    """
    Vaccine uptake model builder.

    `VaxfluxModel` stores the curve, time ranges, covariates, interventions, and
    observations that define a vaccination uptake model. Configure the model by
    incrementally adding these components, then use rendering or sampling methods to
    inspect or estimate the resulting probabilistic model.

    Examples:
        >>> import pandas as pd
        >>> from vaxflux import LogisticCurve, SeasonRange, VaxfluxModel
        >>> observations = pd.DataFrame(
        ...     {
        ...         "season": ["2023/2024"],
        ...         "start_date": ["2023-12-01"],
        ...         "end_date": ["2023-12-07"],
        ...         "type": ["incidence"],
        ...         "value": [0.18],
        ...     }
        ... )
        >>> model = VaxfluxModel(curve=LogisticCurve())
        >>> model = (
        ...     model.add_seasons(
        ...         SeasonRange(
        ...             season="2023/2024",
        ...             start_date="2023-12-01",
        ...             end_date="2024-03-31",
        ...         )
        ...     )
        ...     .add_observations(observations)
        ...     .add_observation_process(
        ...         kind="normal",
        ...         noise=0.05,
        ...     )
        ... )
        >>> model
        <vaxflux.VaxfluxModel object at ...>
    """

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
        self._observations: VaxfluxObservations | None = None
        self._interventions: list[Intervention] = []
        self._implementations: list[Implementation] = []
        self._observation_process_kind: None | Literal["normal"] = None
        self._observation_process_noise: float = 0.05
        self._observation_process_partially_pool_by_season: bool = True
        self._observation_process_partially_pool_by_covariate: bool = False
        self._observation_process_prevalence_penalty: float = 0.0
        self._observation_labels: list[str] = []

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
            >>> from vaxflux import LogisticCurve, SeasonRange
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
            >>> from vaxflux import LogisticCurve, DateRange, SeasonRange
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
            ValueError: If duplicate parameter/covariate pairs are provided.

        Examples:
            >>> from vaxflux._covariates import PartiallyPooledGaussianCovariate
            >>> from vaxflux._curves import LogisticCurve
            >>> model = VaxfluxModel(curve=LogisticCurve())
            >>> model.add_covariates(
            ...     PartiallyPooledGaussianCovariate(
            ...         parameter="m",
            ...         covariate="age",
            ...         mu_mu=0.5,
            ...         mu_sigma=0.1,
            ...         sigma=0.2,
            ...     )
            ... )
            <vaxflux.VaxfluxModel object at ...>
            >>> model.add_covariates(
            ...     [
            ...         PartiallyPooledGaussianCovariate(
            ...             parameter="sigma",
            ...             mu_mu=0.2,
            ...             mu_sigma=0.03,
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
        existing_pairs = Counter(
            (cov.parameter, cov.covariate) for cov in self._covariates
        )
        new_pairs = Counter((cov.parameter, cov.covariate) for cov in covariates)
        duplicate_pairs = [
            pair
            for pair, count in new_pairs.items()
            if count + existing_pairs.get(pair, 0) > 1
        ]
        if duplicate_pairs:
            msg = (
                "Duplicate covariate parameter/covariate pairs found: "
                f"{duplicate_pairs}."
            )
            raise ValueError(msg)
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

    def add_observations(
        self, observations: VaxfluxObservations | pd.DataFrame
    ) -> Self:
        """
        Add observations to the model.

        Args:
            observations: The observations to add to the model. Accepts either a
                `vaxflux.VaxfluxObservations` instance or a raw `pandas.DataFrame`
                which will be validated on construction.

        Returns:
            The model instance for method chaining.

        Raises:
            ValueError: If observations were already added to the model.

        """
        if self._observations is not None:
            msg = "Observations have already been added to the model."
            raise ValueError(msg)
        self._observations = VaxfluxObservations.from_dataframe(observations)
        return self

    def add_observation_process(
        self,
        kind: Literal["normal"],
        noise: float,
        *,
        partially_pool_by_season: bool = True,
        partially_pool_by_covariate: bool = False,
        prevalence_penalty: float = 0.0,
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
            prevalence_penalty: Penalty to apply when modeled prevalence exceeds 1. The
                penalty is applied as `-prevalence_penalty * excess^2` where `excess`
                is the excess prevalence. Use `math.inf` for a hard constraint.

        Returns:
            The model instance for method chaining.

        Raises:
            ValueError: If no observations have been added to the model.
            ValueError: If an unsupported observation process kind is provided.
            ValueError: If the prevalence penalty is negative.

        """
        if self._observations is None:
            msg = "No observations have been added to the model, nothing to observe."
            raise ValueError(msg)
        if kind != "normal":
            msg = f"Unsupported observation process kind: {kind}."
            raise ValueError(msg)
        if prevalence_penalty < 0:
            msg = "prevalence_penalty must be non-negative."
            raise ValueError(msg)
        self._observation_process_kind = kind
        self._observation_process_noise = noise
        self._observation_process_partially_pool_by_season = partially_pool_by_season
        self._observation_process_partially_pool_by_covariate = (
            partially_pool_by_covariate
        )
        self._observation_process_prevalence_penalty = prevalence_penalty
        return self

    def render_model(self, **kwargs: Any) -> Any:  # noqa: ANN401
        """
        Render the model graph using NumPyro's rendering utilities.

        Args:
            **kwargs: Keyword arguments forwarded to `numpyro.render_model`.

        Returns:
            The rendered model object (typically a Graphviz object).

        """
        self._pre_model()
        return _render_model(self._model, **kwargs)

    def sample(  # noqa: PLR0913
        self,
        prior_samples: int | None = None,
        warmup: int | None = None,
        samples: int | None = None,
        chains: int = 1,
        random_seed: int = 1,
        mcmc_args: dict[str, Any] | None = None,
        nuts_args: dict[str, Any] | None = None,
        prior_predictive_args: dict[str, Any] | None = None,
        posterior_predictive_args: dict[str, Any] | None = None,
        arviz_args: dict[str, Any] | None = None,
    ) -> VaxfluxInferenceData:
        """
        Sample from the prior predictive, posterior, and posterior predictive.

        Args:
            prior_samples: Number of prior predictive samples to draw. When `None`,
                prior predictive sampling is skipped.
            warmup: Number of warmup (burn-in) steps.
            samples: Number of posterior samples to draw. When `None`, posterior and
                posterior predictive sampling are skipped.
            chains: Number of MCMC chains to run.
            random_seed: Seed for the random number generator.
            mcmc_args: Additional arguments for the MCMC sampler.
            nuts_args: Additional arguments for the NUTS kernel.
            prior_predictive_args: Additional arguments for the prior predictive
                sampler.
            posterior_predictive_args: Additional arguments for the posterior
                predictive sampler.
            arviz_args: Additional arguments for ArviZ's `from_numpyro`.

        Returns:
            The inference data containing whichever sampling stages were requested.

        """
        if prior_samples is None and samples is None:
            msg = "At least one of `prior_samples` or `samples` must be provided."
            raise ValueError(msg)

        # Prepare arguments
        mcmc_args = mcmc_args or {}
        nuts_args = nuts_args or {}
        prior_predictive_args = prior_predictive_args or {}
        posterior_predictive_args = posterior_predictive_args or {}
        from_numpyro_kwargs = dict(arviz_args or {})

        # Prepare the model
        self._pre_model()
        rng_key = key(random_seed)
        prior_key, mcmc_key, posterior_key = split(rng_key, num=3)
        coords, dims = self._coords_and_dims(
            base_coords=from_numpyro_kwargs.get("coords"),
            base_dims=from_numpyro_kwargs.get("dims"),
        )
        from_numpyro_kwargs["coords"] = coords
        from_numpyro_kwargs["dims"] = dims
        if self._observations is not None:
            from_numpyro_kwargs["observations"] = self._observations.data.copy()

        # Sample from prior predictive if requested
        if prior_samples is not None:
            prior_predictive = Predictive(
                self._model, num_samples=prior_samples, **prior_predictive_args
            )
            from_numpyro_kwargs["prior"] = prior_predictive(
                prior_key, simulate_observations=True
            )

        # Sample from posterior/posterior predictive if requested
        if samples is not None:
            kernel = NUTS(self._model, **nuts_args)
            mcmc = MCMC(
                kernel,
                num_warmup=warmup,
                num_samples=samples,
                num_chains=chains,
                **mcmc_args,
            )
            mcmc_key = split(mcmc_key, num=chains) if chains > 1 else mcmc_key
            mcmc.run(mcmc_key)

            posterior_samples = mcmc.get_samples()
            posterior_predictive = Predictive(
                self._model,
                posterior_samples=posterior_samples,
                **posterior_predictive_args,
            )
            from_numpyro_kwargs["posterior_predictive"] = posterior_predictive(
                posterior_key, simulate_observations=True
            )
            from_numpyro_kwargs["posterior"] = mcmc

        return VaxfluxInferenceData.from_numpyro(**from_numpyro_kwargs)

    def _coords_and_dims(
        self,
        *,
        base_coords: dict[str, list[str]] | None = None,
        base_dims: dict[str, list[str]] | None = None,
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """
        Build ArviZ coordinates and dimensions for model outputs.

        Args:
            base_coords: Optional starting coordinates to extend.
            base_dims: Optional starting dimensions to extend.

        Returns:
            A tuple of (coords, dims) dictionaries.

        """
        coords = dict(base_coords or {})
        dims = dict(base_dims or {})
        for func_short_name in [
            "date_ranges",
            "incidence",
            "days",
            "daily_params",
            "covariates",
            "interventions",
            "observations",
            "observation_sigma",
        ]:
            coord_block, dim_block = getattr(
                self, f"_coords_and_dims_for_{func_short_name}"
            )(coords, dims)
            if coord_block:
                coords.update(coord_block)
            if dim_block:
                dims.update(dim_block)
        return coords, dims

    def _coords_and_dims_for_date_ranges(
        self,
        coords: dict[str, list[str]],
        dims: dict[str, list[str]],  # noqa: ARG002
    ) -> tuple[dict[str, list[str]] | None, dict[str, list[str]] | None]:
        """
        Add date range coordinates.

        Adds per-season date-range coordinates like `date_ranges_{season}` with
        `YYYY-MM-DD_YYYY-MM-DD` labels for each range.

        Args:
            coords: Existing coordinates to extend.
            dims: Existing dimension mappings (unused).

        Returns:
            A `(coords, dims)` tuple where only `coords` is populated.

        """
        new_coords: dict[str, list[str]] = {}
        for season_name, labels in self._date_range_coord_labels.items():
            if not labels:
                continue
            coord_key = self._date_range_coord_keys[season_name]
            if coord_key not in coords:
                new_coords[coord_key] = labels
        return new_coords or None, None

    def _coords_and_dims_for_incidence(
        self,
        coords: dict[str, list[str]],  # noqa: ARG002
        dims: dict[str, list[str]],
    ) -> tuple[dict[str, list[str]] | None, dict[str, list[str]] | None]:
        """
        Add incidence dimensions.

        Maps incidence deterministic values to their per-season date-range coordinates.
        See `_coords_and_dims_for_date_ranges` for formatting of the values, but the
        keys are formatted as "incidence_{season}_{category_combination}".

        Args:
            coords: Existing coordinates (unused).
            dims: Existing dimension mappings to extend.

        Returns:
            A `(coords, dims)` tuple where only `dims` is populated.

        """
        new_dims: dict[str, list[str]] = {}
        for season_name, incidence_names in self._incidence_names_by_season.items():
            if not incidence_names:
                continue
            coord_key = self._date_range_coord_keys[season_name]
            for incidence_name in incidence_names:
                if incidence_name not in dims:
                    new_dims[incidence_name] = [coord_key]
        return None, new_dims or None

    def _coords_and_dims_for_days(
        self,
        coords: dict[str, list[str]],
        dims: dict[str, list[str]],  # noqa: ARG002
    ) -> tuple[dict[str, list[str]] | None, dict[str, list[str]] | None]:
        """
        Add day coordinates.

        Adds per-season coordinates like `days_{season}` with `YYYY-MM-DD` labels
        for each day in the season.

        Args:
            coords: Existing coordinates to extend.
            dims: Existing dimension mappings (unused).

        Returns:
            A `(coords, dims)` tuple where only `coords` is populated.

        """
        new_coords: dict[str, list[str]] = {}
        for season_name, labels in self._days_coord_labels.items():
            if not labels:
                continue
            coord_key = self._days_coord_keys[season_name]
            if coord_key not in coords:
                new_coords[coord_key] = labels
        return new_coords or None, None

    def _coords_and_dims_for_daily_params(
        self,
        coords: dict[str, list[str]],  # noqa: ARG002
        dims: dict[str, list[str]],
    ) -> tuple[dict[str, list[str]] | None, dict[str, list[str]] | None]:
        """
        Add daily parameter dimensions.

        Maps daily parameter deterministic values to the `days_{season}` coordinates.

        Args:
            coords: Existing coordinates (unused).
            dims: Existing dimension mappings to extend.

        Returns:
            A `(coords, dims)` tuple where only `dims` is populated.

        """
        new_dims: dict[str, list[str]] = {}
        for season_name, daily_names in self._daily_param_names_by_season.items():
            if not daily_names:
                continue
            coord_key = self._days_coord_keys[season_name]
            for daily_name in daily_names:
                if daily_name not in dims:
                    new_dims[daily_name] = [coord_key]
        return None, new_dims or None

    def _coords_and_dims_for_covariates(
        self,
        coords: dict[str, list[str]],
        dims: dict[str, list[str]],
    ) -> tuple[dict[str, list[str]] | None, dict[str, list[str]] | None]:
        """
        Add Covariate Coordinates and Dimensions.

        Adds `{covariate}_categories` and `{covariate}_categories_short` coordinates
        and maps covariate draws to those dimensions, with seasonal covariates mapped
        to the `season` coordinate.

        Args:
            coords: Existing coordinates to extend.
            dims: Existing dimension mappings to extend.

        Returns:
            A `(coords, dims)` tuple with any new covariate coordinates and dims.

        """
        new_coords: dict[str, list[str]] = {}
        new_dims: dict[str, list[str]] = {}
        for covariate_name, categories in self._covariate_categories_map.items():
            full_key, short_key = self._covariate_category_coord_keys[covariate_name]
            if full_key not in coords:
                new_coords[full_key] = list(categories)
            if short_key not in coords:
                new_coords[short_key] = list(categories[1:])
        if "covariate_names" not in coords and self._covariate_names:
            new_coords["covariate_names"] = list(self._covariate_names)
        if "season" not in coords:
            new_coords["season"] = list(self._season_names)
        # Derive dimension mappings from the known shape contracts. Seasonal
        # covariates are always shape (num_seasons,) and categorical covariates
        # are always shape (num_seasons, num_categories).
        for covariate in self._covariates:
            self._build_covariate_dims(covariate, dims, new_dims)
        return new_coords or None, new_dims or None

    def _build_covariate_dims(
        self,
        covariate: Covariate,
        existing_dims: dict[str, list[str]],
        new_dims: dict[str, list[str]],
    ) -> None:
        """
        Register ArviZ dimension labels for a single covariate.

        Derives dims for the mandatory `covariate_values_*` deterministic site
        and the raw `{prefix}` sample site from the known shape contracts.
        Any additional sites declared by the covariate via `extra_dims()` are
        also registered here.

        Args:
            covariate: The covariate to register dims for.
            existing_dims: The dims already registered on the model; used to
                avoid overwriting user-supplied overrides.
            new_dims: The mapping to extend with any newly derived dims.

        """
        values_var = f"covariate_values_{covariate.prefix}"
        if values_var in existing_dims:
            return
        raw_var = covariate.prefix
        if covariate.covariate is None:
            new_dims[values_var] = ["season"]
            if raw_var not in existing_dims:
                new_dims[raw_var] = ["season"]
            extra = covariate.extra_dims("season", None)
        else:
            coord_keys = self._covariate_category_coord_keys[covariate.covariate]
            full_key, short_key = coord_keys
            new_dims[values_var] = ["season", full_key]
            if raw_var not in existing_dims:
                new_dims[raw_var] = [short_key]
            extra = covariate.extra_dims("season", short_key)
        new_dims.update(
            {site: dims for site, dims in extra.items() if site not in existing_dims}
        )

    def _coords_and_dims_for_observation_sigma(
        self,
        coords: dict[str, list[str]],
        dims: dict[str, list[str]],
    ) -> tuple[dict[str, list[str]] | None, dict[str, list[str]] | None]:
        """
        Add Observation Sigma Coordinates and Dimensions.

        When partially pooled, maps `observation_sigma` onto `season` and/or
        covariate-category combination coordinates.

        Args:
            coords: Existing coordinates to extend.
            dims: Existing dimension mappings to extend.

        Returns:
            A `(coords, dims)` tuple with any new observation sigma metadata.

        """
        if self._observations is None or self._observation_process_kind is None:
            return None, None
        if "observation_sigma" in dims:
            return None, None
        new_coords: dict[str, list[str]] = {}
        if self._covariate_combo_coord_key not in coords:
            new_coords[self._covariate_combo_coord_key] = self._covariate_combo_labels
        sigma_dims: list[str] = []
        if self._observation_process_partially_pool_by_season:
            sigma_dims.append("season")
        if self._observation_process_partially_pool_by_covariate:
            sigma_dims.append(self._covariate_combo_coord_key)
        new_dims = {"observation_sigma": sigma_dims} if sigma_dims else None
        return new_coords or None, new_dims

    def _coords_and_dims_for_observations(
        self,
        coords: dict[str, list[str]],
        dims: dict[str, list[str]],
    ) -> tuple[dict[str, list[str]] | None, dict[str, list[str]] | None]:
        """
        Add observation coordinates and dimensions.

        Adds an `observation` coordinate with labels describing each observation
        and maps observation-related variables to that dimension.

        Args:
            coords: Existing coordinates to extend.
            dims: Existing dimension mappings to extend.

        Returns:
            A `(coords, dims)` tuple with any new observation metadata.

        """
        if self._observations is None or not self._observation_labels:
            return None, None
        new_coords: dict[str, list[str]] = {}
        new_dims: dict[str, list[str]] = {}
        if "observation" not in coords:
            new_coords["observation"] = list(self._observation_labels)
        if "observation" not in dims:
            new_dims["observation"] = ["observation"]
        if "observation_mean" not in dims:
            new_dims["observation_mean"] = ["observation"]
        if "observation_sim" not in dims:
            new_dims["observation_sim"] = ["observation"]
        return new_coords or None, new_dims or None

    def _coords_and_dims_for_interventions(
        self,
        coords: dict[str, list[str]],
        dims: dict[str, list[str]],
    ) -> tuple[dict[str, list[str]] | None, dict[str, list[str]] | None]:
        """
        Add intervention coordinates and dimensions.

        Maps intervention samples to their implementation dimensions with labels
        like `implementation_0`.

        Args:
            coords: Existing coordinates to extend.
            dims: Existing dimension mappings to extend.

        Returns:
            A `(coords, dims)` tuple with any new intervention metadata.

        """
        new_coords: dict[str, list[str]] = {}
        new_dims: dict[str, list[str]] = {}
        for name, coord_key in self._intervention_impl_coord_keys.items():
            labels = self._intervention_impl_coord_labels.get(name, [])
            if labels and coord_key not in coords:
                new_coords[coord_key] = labels
            intervention_name = _coord_name("intervention", name)
            if intervention_name not in dims and coord_key:
                new_dims[intervention_name] = [coord_key]
        return new_coords or None, new_dims or None

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
                self._observations.data,
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
        self._date_ranges_by_season = {
            season: [
                date_range for date_range in self._dates if date_range.season == season
            ]
            for season in self._season_names
        }
        self._date_range_indices_by_season = {
            season: {
                (date_range.start_date, date_range.end_date): idx
                for idx, date_range in enumerate(date_ranges)
            }
            for season, date_ranges in self._date_ranges_by_season.items()
        }
        self._date_range_coord_keys = {
            season: _coord_name("date_ranges", season) for season in self._season_names
        }
        self._date_range_coord_labels = {
            season: [
                f"{date_range.start_date.isoformat()}_{date_range.end_date.isoformat()}"
                for date_range in date_ranges
            ]
            for season, date_ranges in self._date_ranges_by_season.items()
        }
        self._days_coord_keys = {
            season: _coord_name("days", season) for season in self._season_names
        }
        self._days_coord_labels = {
            season: [
                (season_range.start_date + timedelta(days=offset)).isoformat()
                for offset in range(self._season_day_counts[season])
            ]
            for season, season_range in self._season_map.items()
        }
        self._season_tokens = {
            season_name: _coord_name(season_name) for season_name in self._season_names
        }
        self._covariate_categories_map = {
            category.covariate: category.categories
            for category in self._covariate_categories
        }
        self._covariate_names = list(self._covariate_categories_map.keys())
        _covariate_name_indices = {
            name: idx for idx, name in enumerate(self._covariate_names)
        }
        _covariate_category_indices = {
            name: {category: idx for idx, category in enumerate(categories)}
            for name, categories in self._covariate_categories_map.items()
        }
        self._covariate_index_info = [
            CovariateIndexInfo(
                covariate=cov,
                resolved_categories=list(self._covariate_categories_map[cov.covariate])
                if cov.covariate is not None
                else None,
                combo_position=_covariate_name_indices.get(cov.covariate)
                if cov.covariate is not None
                else None,
                category_index_map=_covariate_category_indices.get(cov.covariate)
                if cov.covariate is not None
                else None,
            )
            for cov in self._covariates
        ]
        if self._observations is not None:
            self._observation_labels = [
                self._format_observation_label(row)
                for _, row in self._observations.data.iterrows()
            ]
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
            param: [
                info
                for info in self._covariate_index_info
                if info.covariate.parameter == param
            ]
            for param in self._curve.parameters
        }
        self._incidence_names_by_season = {
            season: [
                _coord_name(
                    "incidence",
                    season,
                    *self._category_combo_with_name(category_combo),
                )
                for category_combo in self._category_combinations
            ]
            for season in self._season_names
        }
        self._daily_param_names_by_season = {
            season: [
                _coord_name(
                    _coord_name(
                        param,
                        season,
                        *self._category_combo_with_name(category_combo),
                    ),
                    "daily",
                )
                for param in self._curve.parameters
                for category_combo in self._category_combinations
            ]
            for season in self._season_names
        }
        self._covariate_category_coord_keys = {
            covariate_name: (
                _coord_name(covariate_name, "categories"),
                _coord_name(covariate_name, "categories", "short"),
            )
            for covariate_name in self._covariate_categories_map
        }
        self._covariate_combo_coord_key = _coord_name(
            "covariate", "category", "combinations"
        )
        self._covariate_combo_labels = [
            "_".join(parts) if parts else "all"
            for parts in (
                self._category_combo_with_name(combo)
                for combo in self._category_combinations
            )
        ]
        self._interventions_by_name = {
            intervention.name: [
                implementation
                for implementation in self._implementations
                if implementation.intervention == intervention.name
            ]
            for intervention in self._interventions
        }
        self._intervention_impl_coord_keys = {
            name: _coord_name("intervention", name, "implementations")
            for name in self._interventions_by_name
        }
        self._intervention_impl_coord_labels = {
            name: [
                f"implementation_{idx}"
                for idx in range(len(self._interventions_by_name[name]))
            ]
            for name in self._interventions_by_name
        }

    def _model(self, *, simulate_observations: bool = False) -> None:
        """
        Define the model for inference.

        Args:
            simulate_observations: Whether to simulate observations as part of the
                model. Particularly useful for posterior/prior predictive sampling.
        """
        covariate_values = self._model_sample_covariates()
        summed_parameters = self._model_summed_parameters(covariate_values)
        daily_summed_parameters = self._model_daily_summed_parameters(summed_parameters)
        daily_summed_parameters = self._model_apply_interventions(
            daily_summed_parameters
        )
        observations = self._model_observations(daily_summed_parameters)
        self._model_observation_process(
            observations,
            simulate_observations=simulate_observations,
        )

    def _model_sample_covariate(self, info: CovariateIndexInfo) -> jnp.ndarray:
        """
        Sample from a single covariate.

        Args:
            info: Pre-resolved index info for the covariate, built during
                `_pre_model`. Provides the covariate and its already-resolved
                categories so no runtime dict lookup is needed.

        Returns:
            The sampled covariate values.

        """
        sampled_with_context = info.covariate.sample(
            season_names=self._season_names,
            covariate_categories=info.resolved_categories,
        )
        return cast("jnp.ndarray", sampled_with_context)

    def _model_sample_covariates(self) -> dict[str, jnp.ndarray]:
        """
        Sample from all covariates.

        Returns:
            A dictionary mapping covariate prefixes to their sampled values.

        """
        return {
            info.covariate.prefix: self._model_sample_covariate(info)
            for info in self._covariate_index_info
        }

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
                    season_index = self._season_name_indices[season]
                    for info in self._covariates_by_parameter.get(param, []):
                        values = covariate_values[info.covariate.prefix]
                        if info.combo_position is None:
                            # Seasonal covariates: always 1D (num_seasons,).
                            components.append(values[season_index])
                        else:
                            # Categorical covariates: always 2D.
                            category = category_combo[info.combo_position]
                            category_index = info.category_index_map[category]  # type: ignore[index]
                            components.append(values[season_index, category_index])
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
            intervention_name = _coord_name("intervention", intervention.name)
            plate_name = _coord_name(intervention_name, "implementations")
            with numpyro.plate(plate_name, len(intervention_implementations)):
                intervention_values = intervention.sample(intervention_name)
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
            A dictionary mapping season/category names to arrays of incidence values
            aligned to that season's date range order.

        """
        if not self._dates:
            return {}
        incidence_by_date_range: dict[str, jnp.ndarray] = {}
        for season_name, date_ranges in self._date_ranges_by_season.items():
            if not date_ranges:
                continue
            for category_combo in self._category_combinations:
                daily_params: dict[str, jnp.ndarray] = {}
                for param in self._curve.parameters:
                    base_name = _coord_name(
                        param,
                        season_name,
                        *self._category_combo_with_name(category_combo),
                    )
                    daily_name = _coord_name(base_name, "daily")
                    daily_params[param] = daily_summed_parameters[daily_name]
                incidence_values: list[jnp.ndarray] = []
                for date_range in date_ranges:
                    season_range = self._season_map[date_range.season]
                    start_idx = (date_range.start_date - season_range.start_date).days
                    end_idx = (date_range.end_date - season_range.start_date).days
                    date_indices = list(range(start_idx, end_idx + 1))
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
                    incidence_values.append(jnp.sum(daily_incidence))
                incidence_name = _coord_name(
                    "incidence",
                    season_name,
                    *self._category_combo_with_name(category_combo),
                )
                incidence_series = jnp.stack(incidence_values)
                prevalence_penalty = self._observation_process_prevalence_penalty
                if not math.isclose(prevalence_penalty, 0.0):
                    penalty_name = _coord_name(incidence_name, "prevalence_penalty")
                    excess = jnp.maximum(jnp.sum(incidence_series) - 1.0, 0.0)
                    if math.isinf(prevalence_penalty):
                        numpyro.factor(
                            penalty_name, jnp.where(excess > 0.0, -jnp.inf, 0.0)
                        )
                    else:
                        numpyro.factor(
                            penalty_name, -prevalence_penalty * jnp.square(excess)
                        )
                incidence_by_date_range[incidence_name] = numpyro.deterministic(
                    incidence_name,
                    incidence_series,
                )
        return incidence_by_date_range

    def _model_observation_process(
        self,
        observations: dict[str, jnp.ndarray],
        *,
        simulate_observations: bool,
    ) -> None:
        """
        Fit the observation process to the model observations.

        Args:
            observations: A dictionary mapping observation names to incidence values.
            simulate_observations: Whether to sample observation-level draws for
                predictive checks.

        """
        if self._observations is None:
            return
        sigma = self._model_observation_process_sigma()
        obs_means: list[jnp.ndarray] = []
        obs_scales: list[jnp.ndarray] = []
        obs_values: list[float] = []
        for _, row in self._observations.data.iterrows():
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
            )
            if observation_name not in observations:
                msg = f"Observation key '{observation_name}' not found in model output."
                raise ValueError(msg)
            date_range_index = self._date_range_indices_by_season.get(
                row["season"], {}
            ).get((start_date, end_date))
            if date_range_index is None:
                msg = (
                    "Observation date range not found in model output for season "
                    f"'{row['season']}' with dates {start_date} to {end_date}."
                )
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
            obs_means.append(observations[observation_name][date_range_index])
            obs_scales.append(scaled_sigma)
            obs_values.append(float(row["value"]))
        if not obs_means:
            return
        obs_mean_array = jnp.stack(obs_means)
        obs_scale_array = jnp.stack(obs_scales)
        obs_value_array = jnp.asarray(obs_values)
        numpyro.deterministic("observation_mean", obs_mean_array)
        with numpyro.plate("observation_plate", len(obs_values)):
            numpyro.sample(
                "observation",
                numpyro.distributions.Normal(obs_mean_array, obs_scale_array),
                obs=obs_value_array,
            )
            if simulate_observations:
                numpyro.sample(
                    "observation_sim",
                    numpyro.distributions.Normal(obs_mean_array, obs_scale_array),
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

    def _format_observation_label(self, row: pd.Series) -> str:
        start_date = pd.to_datetime(row["start_date"]).date().isoformat()
        end_date = pd.to_datetime(row["end_date"]).date().isoformat()
        parts = [str(row["season"]), start_date, end_date]
        parts.extend(
            f"{covariate_name}={row[covariate_name]}"
            for covariate_name in self._covariate_names
        )
        return "|".join(parts)
