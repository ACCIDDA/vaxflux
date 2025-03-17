"""Unit tests for the `vaxflux._util._clean_name` function."""

import re
from collections.abc import Callable

import pytest

from vaxflux._util import _clean_name


@pytest.mark.parametrize(
    "args",
    (
        ("a", "b", "c"),
        ("a",),
        (None, "a", None),
        (None, None, None),
        ("abc", "de", "f"),
        ("a01", "b02", "c@d", None, "!ef"),
        (None, "abCD", "ef", None, "gHiJkL"),
    ),
)
@pytest.mark.parametrize("joiner", ("", "_", "--"))
@pytest.mark.parametrize(
    "transform",
    (lambda x: x, lambda x: x.lower(), lambda x: x.upper(), lambda x: x.title()),
)
def test_output_validation(
    args: tuple[str, ...], joiner: str, transform: Callable[[str], str]
) -> None:
    """Test that the output is a string."""
    cleaned_name = _clean_name(*args, joiner=joiner, transform=transform)
    assert isinstance(cleaned_name, str)
    for arg in args:
        if arg is not None and re.match(r"^[a-zA-Z0-9]+$", arg):
            assert transform(arg) in cleaned_name
    if sum(arg is not None for arg in args) > 1:
        assert joiner in cleaned_name
    assert cleaned_name == cleaned_name.strip()
