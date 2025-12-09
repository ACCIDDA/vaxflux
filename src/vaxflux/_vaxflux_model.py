__all__: list[str] = []

from typing import Any, Self

import numpyro
from jax.random import key
from numpyro.infer import MCMC, NUTS

from vaxflux._covariates import Covariate
from vaxflux._curves import Curve
from vaxflux._util import _collect_args
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
        self._kernel = NUTS(self._model, **nuts_args)
        self._mcmc = MCMC(self._kernel, **mcmc_args)
        self._rng_key = key(random_seed)
        with numpyro.handlers.seed(numpyro.handlers.trace(self._model), self._rng_key):
            self._mcmc.run(self._rng_key, num_warmup=warmup, num_samples=samples)

    def _model(self) -> None:
        raise NotImplementedError
