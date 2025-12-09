"""Unit tests for the `vaxflux._util._collect_args` function."""

import pytest

from vaxflux._util import _collect_args


@pytest.mark.parametrize(
    ("args_factory", "expected_count"),
    [
        # Single object
        (lambda: (1,), 1),
        # Multiple objects as args
        (lambda: (1, 2, 3), 3),
        # List of objects
        (lambda: ([1, 2, 3],), 3),
        # Tuple of objects
        (lambda: ((1, 2, 3),), 3),
        # Mixed args and sequences
        (lambda: (1, [2, 3], 4), 4),
        # Multiple sequences
        (lambda: ([1, 2], [3, 4]), 4),
        # Mixed single and multiple sequences
        (lambda: (1, [2, 3], 4, [5, 6, 7]), 7),
        # Empty call
        (lambda: (), 0),
        # Empty list
        (lambda: ([],), 0),
        # Empty tuple
        (lambda: ((),), 0),
        # Multiple empty sequences
        (lambda: ([], (), []), 0),
        # Mix of empty and non-empty sequences
        (lambda: ([], 1, (), 2, []), 2),
    ],
)
def test_input_variations(args_factory: object, expected_count: int) -> None:
    """Test collecting objects with various input patterns."""
    args = args_factory()  # type: ignore[operator]
    result = _collect_args(args, int, "int")

    # Check expected number of objects were collected
    assert len(result) == expected_count
    # Check all items are integers
    assert all(isinstance(item, int) for item in result)
    # Check result is a list
    assert isinstance(result, list)


def test_preserves_order() -> None:
    """Test that _collect_args preserves the order of items."""
    args = (3, [1, 2], 5, [4, 6], 7)
    result = _collect_args(args, int, "int")

    assert result == [3, 1, 2, 5, 4, 6, 7]


def test_with_different_types() -> None:
    """Test that _collect_args works with different object types."""

    class TestClass:
        def __init__(self, value: int) -> None:
            self.value = value

    obj1 = TestClass(1)
    obj2 = TestClass(2)
    obj3 = TestClass(3)

    args = (obj1, [obj2, obj3])
    result = _collect_args(args, TestClass, "TestClass")

    assert len(result) == 3
    assert all(isinstance(item, TestClass) for item in result)
    assert result[0].value == 1
    assert result[1].value == 2
    assert result[2].value == 3


@pytest.mark.parametrize(
    "invalid_arg", ["not an int", 12.5, {"key": "value"}, None, object()]
)
def test_invalid_type_error(invalid_arg: object) -> None:
    """Test TypeError is raised when passing invalid types."""
    with pytest.raises(
        TypeError,
        match=r"Arguments must be int objects or sequences of int objects, got \w+\.",
    ):
        _collect_args((invalid_arg,), int, "int")


@pytest.mark.parametrize("invalid_item", ["not an int", 12.5, {"key": "value"}])
def test_sequence_with_invalid_items(invalid_item: object) -> None:
    """Test TypeError is raised when a sequence contains invalid items."""
    with pytest.raises(
        TypeError,
        match=r"All items in a sequence must be int objects, got \w+\.",
    ):
        _collect_args(([1, 2, invalid_item],), int, "int")  # type: ignore[list-item]


@pytest.mark.parametrize("word", ["abc", b"abc"])
def test_string_and_bytes_not_treated_as_sequence(word: str | bytes) -> None:
    """Test that strings are not treated as sequences even though they're iterable."""
    with pytest.raises(
        TypeError,
        match=r"Arguments must be int objects or sequences of int objects, got \w+\.",
    ):
        _collect_args((word,), int, "int")


def test_mixed_valid_and_invalid() -> None:
    """Test that an error is raised even when some items are valid."""
    with pytest.raises(TypeError):
        _collect_args((1, 2, "invalid", 3), int, "int")


def test_error_in_middle_of_sequence() -> None:
    """Test an error is raised when an invalid item is in the middle of a sequence."""
    with pytest.raises(
        TypeError,
        match=r"All items in a sequence must be int objects, got str\.",
    ):
        _collect_args(([1, 2, "invalid", 4, 5],), int, "int")  # type: ignore[list-item]


def test_custom_type_name_in_error() -> None:
    """Test that the custom type name appears in error messages."""

    class CustomClass:
        pass

    with pytest.raises(
        TypeError,
        match=(
            r"Arguments must be CustomType objects or sequences of CustomType objects"
        ),
    ):
        _collect_args(("invalid",), CustomClass, "CustomType")


def test_nested_sequences() -> None:
    """Test that nested sequences are flattened one level."""
    # Only the outer sequence should be unpacked, not nested ones
    inner_list = [1, 2]
    args = ([inner_list, 3],)

    # This should raise an error because inner_list is not an int
    with pytest.raises(
        TypeError,
        match=r"All items in a sequence must be int objects, got list\.",
    ):
        _collect_args(args, int, "int")  # type: ignore[arg-type]


def test_with_range() -> None:
    """Test that range objects work as sequences."""
    args = (range(1, 4),)
    result = _collect_args(args, int, "int")
    assert result == [1, 2, 3]
