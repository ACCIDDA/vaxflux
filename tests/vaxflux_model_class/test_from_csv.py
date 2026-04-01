"""Unit tests for `VaxfluxModel.from_csv`."""

from datetime import date
from pathlib import Path

import pandas as pd

from vaxflux import LogisticCurve, SeasonVaryingPartiallyPooledGaussianCovariate
from vaxflux._vaxflux_model import VaxfluxModel
from vaxflux._vaxflux_observations import VaxfluxObservations


def test_from_csv_builds_default_model_from_observations(tmp_path: Path) -> None:
    """`from_csv` builds the requested default model configuration."""
    csv_path = tmp_path / "observations.psv"
    pd.DataFrame(
        data={
            "season": ["2024/2025", "2023/2024", "2023/2024"],
            "start_date": ["2024-10-01", "2023-10-01", "2023-10-08"],
            "end_date": ["2024-10-07", "2023-10-07", "2023-10-14"],
            "type": ["incidence", "incidence", "incidence"],
            "value": [0.3, 0.1, 0.2],
            "age": ["50+", "18-49", "50+"],
            "region": ["west", "west", "east"],
        }
    ).to_csv(csv_path, sep="|", index=False)

    model = VaxfluxModel.from_csv(csv_path, sep="|")

    assert isinstance(model._curve, LogisticCurve)
    assert isinstance(model._observations, VaxfluxObservations)
    assert model._interventions == []
    assert model._implementations == []
    assert model._observation_process_kind == "normal"
    assert model._observation_process_noise == 0.001
    assert [
        (season.season, season.start_date, season.end_date) for season in model._seasons
    ] == [
        ("2023/2024", date(2023, 10, 1), date(2023, 10, 14)),
        ("2024/2025", date(2024, 10, 1), date(2024, 10, 7)),
    ]
    assert [
        (
            date_range.season,
            date_range.start_date,
            date_range.end_date,
            date_range.report_date,
        )
        for date_range in model._dates
    ] == [
        ("2023/2024", date(2023, 10, 1), date(2023, 10, 7), date(2023, 10, 7)),
        ("2023/2024", date(2023, 10, 8), date(2023, 10, 14), date(2023, 10, 14)),
        ("2024/2025", date(2024, 10, 1), date(2024, 10, 7), date(2024, 10, 7)),
    ]
    assert {
        categories.covariate: categories.categories
        for categories in model._covariate_categories
    } == {
        "age": ("18-49", "50+"),
        "region": ("east", "west"),
    }
    assert len(model._covariates) == len(model._curve.parameters) * 2
    assert all(
        isinstance(covariate, SeasonVaryingPartiallyPooledGaussianCovariate)
        for covariate in model._covariates
    )
    typed_covariates = [
        covariate
        for covariate in model._covariates
        if isinstance(covariate, SeasonVaryingPartiallyPooledGaussianCovariate)
    ]
    assert len(typed_covariates) == len(model._covariates)
    assert {
        (covariate.parameter, covariate.covariate) for covariate in model._covariates
    } == {
        ("m", "age"),
        ("r", "age"),
        ("s", "age"),
        ("m", "region"),
        ("r", "region"),
        ("s", "region"),
    }
    assert all(covariate.mu_mu == 0.0 for covariate in typed_covariates)
    assert all(covariate.mu_sigma == 1.0 for covariate in typed_covariates)
    assert all(covariate.sigma == 1.0 for covariate in typed_covariates)
