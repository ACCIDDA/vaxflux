"""Unit tests for `vaxflux.covariates.CovariateCategories`."""

import pytest

from vaxflux.covariates import CovariateCategories


@pytest.mark.parametrize(
    ("categories", "duplicate_category"),
    [
        (("a", "b", "c", "c"), "c"),
        (("a", "b", "c", "a"), "a"),
        (("a", "b", "c", "b"), "b"),
        (("a", "b", "a", "b"), "a"),
    ],
)
def test_nonunique_categories_value_error(
    categories: tuple[str],
    duplicate_category: str,
) -> None:
    """Non-unique categories raise a value error."""
    assert len(set(categories)) != len(categories)
    with pytest.raises(
        ValueError,
        match=f"Category '{duplicate_category}' is not unique.",
    ):
        CovariateCategories(covariate="test", categories=categories)
