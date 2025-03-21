"""Unit tests for the `vaxflux._util.ListOfFloats` pydantic type."""

import pytest
from pydantic import BaseModel

from vaxflux._util import ListOfFloats


class ListOfFloatsTestModel(BaseModel):
    """
    Test model for the `ListOfFloats` pydantic type.

    Attributes:
        values: The list of floats.
    """

    values: ListOfFloats


@pytest.mark.parametrize(
    ("values", "expected"),
    (
        ([1.2, 3], [1.2, 3]),
        ([1.2, 3.0], [1.2, 3.0]),
        ([1, 3], [1.0, 3.0]),
        (3, [3.0]),
        (1.2, [1.2]),
    ),
)
def test_output_validation(
    values: float | int | list[float | int], expected: list[float]
) -> None:
    """Test that the output is a list of floats."""
    model = ListOfFloatsTestModel(values=values)  # type: ignore[arg-type]
    assert isinstance(model.values, list)
    assert all(isinstance(v, float) for v in model.values)
    assert model.values == expected
