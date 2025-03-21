"""Unit tests for the `vaxflux._util._pm_name` function."""

import re

import pytest

from vaxflux._util import _pm_name

PM_NAME_REGEX = re.compile(r"^(?:[A-Z]{1}([a-z0-9]+)?)+$")


@pytest.mark.parametrize(
    "args",
    (
        ("m", "age", "18-45yr"),
        ("nu", "county", "wake"),
        ("m", "age", "18-45yr", None),
        ("qrs", None, "18-45yr"),
        ("season", "2022-2023", "dates"),
        ("season", "Winter 22'", "dates"),
        ("covariate", "age", "none"),
    ),
)
def test_output_validation(args: tuple[str | None, ...]) -> None:
    """Test that the output is a string of the expected format."""
    pm_name = _pm_name(*args)
    assert isinstance(pm_name, str)
    assert PM_NAME_REGEX.match(pm_name) is not None
    if "none" not in {a.lower() for a in args if isinstance(a, str)} and None in args:
        assert "None" not in pm_name
