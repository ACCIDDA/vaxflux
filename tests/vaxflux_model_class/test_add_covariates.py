"""Unit tests for the `VaxfluxModel.add_covariates` method."""

import pytest

from vaxflux import (
    Covariate,
    LogisticCurve,
    PartiallyPooledGaussianCovariate,
    VaxfluxModel,
)


@pytest.mark.parametrize(
    ("existing_covariates", "new_covariates", "match_pattern"),
    [
        (
            None,
            [
                PartiallyPooledGaussianCovariate(
                    parameter="m",
                    covariate="age",
                    mu_mu=0.5,
                    mu_sigma=0.1,
                    sigma=0.2,
                ),
                PartiallyPooledGaussianCovariate(
                    parameter="m",
                    covariate="age",
                    mu_mu=0.6,
                    mu_sigma=0.1,
                    sigma=0.3,
                ),
            ],
            r"\[\('m', 'age'\)\]\.$",
        ),
        (
            [
                PartiallyPooledGaussianCovariate(
                    parameter="r",
                    covariate="sex",
                    mu_mu=0.5,
                    mu_sigma=0.1,
                    sigma=0.2,
                ),
            ],
            [
                PartiallyPooledGaussianCovariate(
                    parameter="r",
                    covariate="sex",
                    mu_mu=0.6,
                    mu_sigma=0.1,
                    sigma=0.3,
                ),
            ],
            r"\[\('r', 'sex'\)\]\.$",
        ),
    ],
)
def test_add_covariates_duplicate_pairs(
    existing_covariates: list[Covariate] | None,
    new_covariates: list[Covariate],
    match_pattern: str,
) -> None:
    """
    Adding duplicate covariate pairs raises a `ValueError`.

    Args:
        existing_covariates: Existing covariates to add to the model.
        new_covariates: New covariates to add to the model.
        match_pattern: Regex pattern to match in the ValueError message.

    """
    model = VaxfluxModel(curve=LogisticCurve())

    if existing_covariates:
        model.add_covariates(*existing_covariates)

    match_prefix = r"^Duplicate covariate parameter/covariate pairs found: "
    with pytest.raises(ValueError, match=match_prefix + match_pattern):
        model.add_covariates(*new_covariates)
