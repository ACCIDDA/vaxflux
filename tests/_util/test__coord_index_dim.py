"""Unit tests for the `vaxflux._util._coord_index_dim` function."""

import pytest

from vaxflux._util import _coord_index_dim, _coord_name

EXAMPLE_COORDS = {
    "covariate_age_categories": ["youth", "adult", "senior"],
    "covariate_age_categories_limited": ["adult", "senior"],
    "covariate_names": ["sex", "age"],
    "covariate_sex_categories": ["female", "male"],
    "covariate_sex_categories_limited": ["male"],
    "season": ["2022/2023", "2023/2024"],
}
example_args = []
for dim, values in EXAMPLE_COORDS.items():
    if dim == "season":
        continue
    for i, value in enumerate(values):
        covariate_name = (
            dim.split("_")[1]
            if (dim.startswith("covariate_") and dim != "covariate_names")
            else value
        )
        example_args.append((dim, covariate_name, value, i))


@pytest.mark.parametrize("dim", ("unknown", "nope", "nada"))
@pytest.mark.parametrize("covariate_name", ("age", "sex"))
def test_unknown_dimension_not_implemented_error(dim: str, covariate_name: str) -> None:
    """Test that an unknown dimension raises a NotImplementedError."""
    assert dim not in {
        "season",
        "covariate_names",
        _coord_name("covariate", covariate_name, "categories"),
        _coord_name("covariate", covariate_name, "categories", "limited"),
    }
    with pytest.raises(NotImplementedError, match=f"Unknown dimension: '{dim}'."):
        _coord_index_dim(dim, "2023/2024", "age", "adult", EXAMPLE_COORDS)


@pytest.mark.parametrize("season", EXAMPLE_COORDS.get("season", []))
@pytest.mark.parametrize(
    ("dim", "covariate_name", "category", "expected"), example_args
)
def test_exact_values_for_select_inputs_independent_of_season(
    dim: str,
    season: str,
    covariate_name: str | None,
    category: str | None,
    expected: int,
) -> None:
    """Test the exact output value for select inputs."""
    assert (
        _coord_index_dim(dim, season, covariate_name, category, EXAMPLE_COORDS)
        == expected
    )
