__all__: list[str] = []

import itertools
from itertools import chain
from typing import Any, Self, cast

import jax.numpy as jnp
import numpyro
import pandas as pd
from jax import Array as JaxArray
from jax.random import key
from numpyro.infer import MCMC, NUTS, Predictive

from vaxflux._covariates import Covariate
from vaxflux._curves import Curve
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
            >>> result = model.add_seasons(
            ...     SeasonRange(
            ...         season="2023/2024",
            ...         start_date="2023-12-01",
            ...         end_date="2024-03-31",
            ...     )
            ... )
            >>> result = model.add_seasons(
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
            >>> result = model.add_seasons(
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
            >>> from vaxflux.dates import DateRange
            >>> model = VaxfluxModel(curve=LogisticCurve())
            >>> result = model.add_dates(
            ...     DateRange(
            ...         season="2023/2024",
            ...         start_date="2023-12-01",
            ...         end_date="2023-12-07",
            ...         report_date="2023-12-08",
            ...     )
            ... )
            >>> model.add_dates("invalid_argument")
            Traceback (most recent call last):
                ...
            TypeError: Arguments must be DateRange objects or sequences of DateRange objects, got str.

        """  # noqa: E501
        dates = _collect_ranges(args, DateRange, "DateRange")
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
            >>> result = model.add_covariates(
            ...     PartiallyPooledGaussianCovariate(
            ...         parameter="m",
            ...         covariate="age",
            ...         mu=(0.5, 0.1),
            ...         sigma=0.2,
            ...     )
            ... )
            >>> result = model.add_covariates(
            ...     [
            ...         PartiallyPooledGaussianCovariate(
            ...             parameter="sigma",
            ...             mu=(0.2, 0.03),
            ...             sigma=0.1,
            ...         ),
            ...     ]
            ... )
            >>> model.add_covariates("invalid_argument")
            Traceback (most recent call last):
                ...
            TypeError: Arguments must be Covariate objects or sequences of Covariate objects, got str.

        """  # noqa: E501
        covariates = _collect_args(args, Covariate, "Covariate")  # type: ignore[type-abstract]
        self._covariates.extend(covariates)
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
        self._season_names = [season.season for season in self._seasons]
        self._season_name_indices = {
            name: idx for idx, name in enumerate(self._season_names)
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
        self._covariates_by_parameter = {
            param: [cov for cov in self._covariates if cov.parameter == param]
            for param in self._curve.parameters
        }

    def _model(self) -> None:
        """Define the model for inference."""
        covariate_values = self._model_sample_covariates()
        self._model_summed_parameters(covariate_values)

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
