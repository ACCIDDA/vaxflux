"""Unit tests for the shape contracts of concrete `Covariate` subclasses."""

import jax
import pytest
from numpyro.infer import Predictive

from vaxflux import (
    Covariate,
    PartiallyPooledGaussianCovariate,
    PooledGaussianCovariate,
    SeasonVaryingPartiallyPooledGaussianCovariate,
)


def _sample_shape(
    covariate: Covariate,
    season_names: list[str],
    covariate_categories: list[str] | None,
) -> tuple[int, ...]:
    """Run a covariate's `sample()` inside a numpyro model and return the shape."""

    def model() -> None:
        covariate.sample(
            season_names=season_names,
            covariate_categories=covariate_categories,
        )

    predictive = Predictive(model, num_samples=1)
    samples = predictive(jax.random.key(0))
    key = f"covariate_values_{covariate.prefix}"
    # Strip the leading sample dimension added by Predictive.
    return tuple(samples[key].shape[1:])


@pytest.mark.parametrize(
    ("covariate", "season_names", "covariate_categories", "expected_shape"),
    [
        (
            PooledGaussianCovariate(parameter="m", mu=0.0, sigma=1.0),
            ["2023/2024", "2024/2025"],
            None,
            (2,),
        ),
        (
            PooledGaussianCovariate(parameter="m", covariate="age", mu=0.0, sigma=1.0),
            ["2023/2024", "2024/2025"],
            ["baseline", "youth", "adult"],
            (2, 3),
        ),
        (
            PartiallyPooledGaussianCovariate(
                parameter="m", mu_mu=0.0, mu_sigma=1.0, sigma=1.0
            ),
            ["2023/2024", "2024/2025"],
            None,
            (2,),
        ),
        (
            PartiallyPooledGaussianCovariate(
                parameter="m", covariate="sex", mu_mu=0.0, mu_sigma=1.0, sigma=1.0
            ),
            ["2023/2024", "2024/2025"],
            ["baseline", "female"],
            (2, 2),
        ),
        (
            SeasonVaryingPartiallyPooledGaussianCovariate(
                parameter="m", covariate="sex", mu_mu=0.0, mu_sigma=1.0, sigma=1.0
            ),
            ["2023/2024", "2024/2025"],
            ["baseline", "female"],
            (2, 2),
        ),
        (
            SeasonVaryingPartiallyPooledGaussianCovariate(
                parameter="m", covariate="age", mu_mu=0.0, mu_sigma=1.0, sigma=1.0
            ),
            ["2022/2023", "2023/2024", "2024/2025"],
            ["baseline", "youth", "adult"],
            (3, 3),
        ),
    ],
)
def test_covariate_sample_shape(
    covariate: Covariate,
    season_names: list[str],
    covariate_categories: list[str] | None,
    expected_shape: tuple[int, ...],
) -> None:
    """Each concrete covariate returns `covariate_values_*` with the expected shape."""
    shape = _sample_shape(covariate, season_names, covariate_categories)
    assert shape == expected_shape


def test_season_varying_requires_covariate_name() -> None:
    """`SeasonVaryingPartiallyPooledGaussianCovariate` requires a covariate name."""
    with pytest.raises(ValueError, match="Input should be a valid string"):
        SeasonVaryingPartiallyPooledGaussianCovariate(
            parameter="m",
            covariate=None,  # type: ignore[arg-type]
            mu_mu=0.0,
            mu_sigma=1.0,
            sigma=1.0,
        )


def test_pooled_categorical_effect_constant_across_seasons() -> None:
    """`PooledGaussianCovariate` categorical effects are constant across seasons."""
    covariate = PooledGaussianCovariate(
        parameter="m", covariate="age", mu=0.5, sigma=0.01
    )

    def model() -> None:
        covariate.sample(
            season_names=["2023/2024", "2024/2025"],
            covariate_categories=["baseline", "youth", "adult"],
        )

    predictive = Predictive(model, num_samples=1)
    samples = predictive(jax.random.key(0))
    values = samples[f"covariate_values_{covariate.prefix}"][0]
    # Shape should be (2, 3): 2 seasons, 3 categories.
    assert values.shape == (2, 3)
    # Both seasons must be identical since the effect is broadcast.
    assert (values[0] == values[1]).all()


def test_partially_pooled_categorical_effect_constant_across_seasons() -> None:
    """`PartiallyPooledGaussianCovariate` categorical effects are constant across seasons."""  # noqa: E501
    covariate = PartiallyPooledGaussianCovariate(
        parameter="m", covariate="age", mu_mu=0.5, mu_sigma=0.01, sigma=0.01
    )

    def model() -> None:
        covariate.sample(
            season_names=["2023/2024", "2024/2025"],
            covariate_categories=["baseline", "youth", "adult"],
        )

    predictive = Predictive(model, num_samples=1)
    samples = predictive(jax.random.key(0))
    values = samples[f"covariate_values_{covariate.prefix}"][0]
    assert values.shape == (2, 3)
    assert (values[0] == values[1]).all()
