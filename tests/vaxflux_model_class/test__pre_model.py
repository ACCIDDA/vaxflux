"""Unit tests for the `VaxfluxModel._pre_model` method."""

from collections.abc import Callable
from datetime import date
from typing import Any

import jax.numpy as jnp
import pytest

from vaxflux import (
    CovariateCategories,
    Curve,
    PartiallyPooledGaussianCovariate,
    SeasonRange,
    VaxfluxModel,
)
from vaxflux._types import NumericalArrayLike


class DummyCurve(Curve):
    """Simple dummy curve for testing."""

    parameters = ("m", "r")

    def prevalence(  # type: ignore[override]
        self, t: NumericalArrayLike, m: NumericalArrayLike, r: NumericalArrayLike
    ) -> NumericalArrayLike:
        """Compute the prevalence at time `t`."""
        return jnp.clip(m * (t - r), 0.0, 1.0)


@pytest.mark.parametrize(
    ("model_factory", "expected"),
    [
        (
            lambda: (
                VaxfluxModel(curve=DummyCurve())
                .add_seasons(
                    SeasonRange(
                        season="2023/2024",
                        start_date=date(2023, 12, 1),
                        end_date=date(2024, 3, 31),
                    ),
                    SeasonRange(
                        season="2024/2025",
                        start_date=date(2024, 12, 1),
                        end_date=date(2025, 3, 31),
                    ),
                )
                .add_covariate_categories(
                    CovariateCategories(covariate="sex", categories=("female", "male"))
                )
                .add_covariates(
                    PartiallyPooledGaussianCovariate(
                        parameter="m",
                        covariate=None,
                        mu_mu=0.0,
                        mu_sigma=1.0,
                        sigma=1.0,
                    ),
                    PartiallyPooledGaussianCovariate(
                        parameter="m",
                        covariate="sex",
                        mu_mu=0.0,
                        mu_sigma=1.0,
                        sigma=1.0,
                    ),
                    PartiallyPooledGaussianCovariate(
                        parameter="r",
                        covariate=None,
                        mu_mu=0.0,
                        mu_sigma=1.0,
                        sigma=1.0,
                    ),
                )
            ),
            {
                "season_names": ["2023/2024", "2024/2025"],
                "season_name_indices": {"2023/2024": 0, "2024/2025": 1},
                "covariate_categories_map": {"sex": ("female", "male")},
                "covariate_names": ["sex"],
                "covariate_name_indices": {"sex": 0},
                "covariate_category_indices": {"sex": {"female": 0, "male": 1}},
                "category_combinations": [("female",), ("male",)],
                "covariates_by_parameter": {
                    "m": ["m_season", "m_sex"],
                    "r": ["r_season"],
                },
            },
        ),
        (
            lambda: (
                VaxfluxModel(curve=DummyCurve())
                .add_seasons(
                    SeasonRange(
                        season="2023/2024",
                        start_date=date(2023, 12, 1),
                        end_date=date(2024, 3, 31),
                    ),
                )
                .add_covariates(
                    PartiallyPooledGaussianCovariate(
                        parameter="m",
                        covariate=None,
                        mu_mu=0.0,
                        mu_sigma=1.0,
                        sigma=1.0,
                    ),
                )
            ),
            {
                "season_names": ["2023/2024"],
                "season_name_indices": {"2023/2024": 0},
                "covariate_categories_map": {},
                "covariate_names": [],
                "covariate_name_indices": {},
                "covariate_category_indices": {},
                "category_combinations": [()],
                "covariates_by_parameter": {"m": ["m_season"], "r": []},
            },
        ),
    ],
)
def test_sets_expected_attributes(
    model_factory: Callable[[], VaxfluxModel], expected: dict[str, Any]
) -> None:
    """Test that `_pre_model` sets the expected attributes."""
    model = model_factory()
    model._pre_model()
    assert model._season_names == expected["season_names"]
    assert model._season_name_indices == expected["season_name_indices"]
    assert model._covariate_categories_map == expected["covariate_categories_map"]
    assert model._covariate_names == expected["covariate_names"]
    assert model._covariate_name_indices == expected["covariate_name_indices"]
    assert model._covariate_category_indices == expected["covariate_category_indices"]
    assert model._category_combinations == expected["category_combinations"]
    assert {
        k: [v.prefix for v in model._covariates_by_parameter.get(k, [])]
        for k in model._covariates_by_parameter
    } == expected["covariates_by_parameter"]
