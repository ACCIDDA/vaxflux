__all__: list[str] = []

from collections.abc import Sequence
from typing import Any, Self

import numpyro
from jax.random import PRNGKey
from numpyro.infer import MCMC, NUTS

from vaxflux._curves import Curve
from vaxflux.dates import SeasonRange, _seasons_overlap


class VaxfluxModel:
    def __init__(self, curve: Curve) -> None:
        """
        Initialize the `VaxfluxModel` with a given uptake curve.

        Args:
            curve: An instance of a `Curve` subclass representing the uptake curve.

        """
        self._curve = curve
        self._seasons: list[SeasonRange] = []

    def add_seasons(self, *args: SeasonRange | Sequence[SeasonRange]) -> Self:
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
        seasons: list[SeasonRange] = []
        for arg in args:
            if isinstance(arg, SeasonRange):
                seasons.append(arg)
            elif isinstance(arg, Sequence) and not isinstance(arg, (str, bytes)):
                # Validate that all items in the sequence are SeasonRange objects
                for item in arg:
                    if not isinstance(item, SeasonRange):
                        msg = (
                            f"All items in a sequence must be SeasonRange objects, "
                            f"got {type(item).__name__}."
                        )
                        raise TypeError(msg)
                seasons.extend(arg)
            else:
                msg = (
                    f"Arguments must be SeasonRange objects or sequences of "
                    f"SeasonRange objects, got {type(arg).__name__}."
                )
                raise TypeError(msg)
        self._validate_and_add_seasons(seasons)
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
        self._rng_key = PRNGKey(random_seed)
        with numpyro.handlers.seed(numpyro.handlers.trace(self._model), self._rng_key):
            self._mcmc.run(self._rng_key, num_warmup=warmup, num_samples=samples)

    def _model(self) -> None:
        raise NotImplementedError

    def _validate_and_add_seasons(self, seasons: list[SeasonRange]) -> None:
        """Validate and add seasons to the model.

        Args:
            seasons: List of seasons to validate and add.

        Raises:
            ValueError: If duplicate season names are found.
            ValueError: If overlapping date ranges are found.
        """
        # Check for duplicate season names within the new seasons
        new_season_names = [season.season for season in seasons]
        if len(new_season_names) != len(set(new_season_names)):
            duplicates = {
                name for name in new_season_names if new_season_names.count(name) > 1
            }
            msg = f"Duplicate season names found in new seasons: {sorted(duplicates)}."
            raise ValueError(msg)

        # Check for duplicate season names against existing seasons
        existing_season_names = {season.season for season in self._seasons}
        conflicting_names = existing_season_names & set(new_season_names)
        if conflicting_names:
            msg = (
                f"Season names already exist in the model: {sorted(conflicting_names)}."
            )
            raise ValueError(msg)

        # Check for overlapping date ranges within new seasons
        for i, season1 in enumerate(seasons):
            for season2 in seasons[i + 1 :]:
                if _seasons_overlap(season1, season2):
                    msg = (
                        f"Overlapping date ranges found: "
                        f"'{season1.season}' ({season1.start_date} to "
                        f"{season1.end_date}) and '{season2.season}' "
                        f"({season2.start_date} to {season2.end_date})."
                    )
                    raise ValueError(msg)

        # Check for overlapping date ranges against existing seasons
        for existing_season in self._seasons:
            for new_season in seasons:
                if _seasons_overlap(existing_season, new_season):
                    msg = (
                        f"Overlapping date ranges found: "
                        f"'{existing_season.season}' ({existing_season.start_date} "
                        f"to {existing_season.end_date}) and '{new_season.season}' "
                        f"({new_season.start_date} to {new_season.end_date})."
                    )
                    raise ValueError(msg)

        # All validations passed, add the seasons
        self._seasons.extend(seasons)
