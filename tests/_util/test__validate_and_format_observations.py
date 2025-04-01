"""Unit tests for `vaxflux._util._validate_and_format_observations`."""

from datetime import date
from itertools import chain, combinations
from typing import Final

import pandas as pd
import pytest
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype, is_string_dtype

from vaxflux._util import (
    _REQUIRED_OBSERVATION_COLUMNS,
    _validate_and_format_observations,
)

PERFECT_OBSERVATIONS: Final = pd.DataFrame.from_records(
    [
        {
            "season": "2022 thru 2023",
            "start_date": date(2022, 1, 1),
            "end_date": date(2022, 1, 31),
            "report_date": date(2022, 2, 1),
            "type": "incidence",
            "value": 0.2,
        },
        {
            "season": "2022 thru 2023",
            "start_date": date(2022, 2, 1),
            "end_date": date(2022, 2, 28),
            "report_date": date(2022, 3, 1),
            "type": "incidence",
            "value": 0.3,
        },
    ],
)


def test_none_returns_none() -> None:
    """Providing `None` returns `None`."""
    assert _validate_and_format_observations(None) is None


@pytest.mark.parametrize(
    "observations",
    [
        pd.DataFrame(),
        pd.DataFrame(data={"incidence": []}),
    ],
)
def test_zero_length_data_frame_value_error(observations: pd.DataFrame) -> None:
    """Providing a zero-length data frame raises a `ValueError`."""
    with pytest.raises(ValueError, match="^No observations provided.$"):
        _validate_and_format_observations(observations)


@pytest.mark.parametrize(
    "columns_to_drop",
    [
        list(cols)
        for cols in chain.from_iterable(
            combinations(_REQUIRED_OBSERVATION_COLUMNS, r)
            for r in range(1, len(_REQUIRED_OBSERVATION_COLUMNS))
        )
    ],
)
def test_missing_required_columns_raises_value_error(
    columns_to_drop: list[str],
) -> None:
    """Providing a DataFrame missing required columns raises a `ValueError`."""
    observations = PERFECT_OBSERVATIONS.drop(columns=columns_to_drop)
    with pytest.raises(
        ValueError,
        match="^The observations DataFrame is missing required columns: .*.$",
    ):
        _validate_and_format_observations(observations)


def test_na_in_value_column_raises_value_error() -> None:
    """Providing a DataFrame with `NaN` in the `value` column raises a `ValueError`."""
    observations = pd.DataFrame(
        data={
            "season": ["2022 thru 2023"],
            "start_date": ["2022-01-01"],
            "end_date": ["2022-01-31"],
            "report_date": ["2022-02-01"],
            "type": ["incidence"],
            "value": [None],
        }
    )
    with pytest.raises(
        ValueError,
        match=(
            "^The observations DataFrame contains "
            "invalid values in the 'value' column.$"
        ),
    ):
        _validate_and_format_observations(observations)


def test_value_column_contains_negative_values_raises_value_error() -> None:
    """Observations with negative values in the `value` column raises a `ValueError`."""
    observations = pd.DataFrame(
        data={
            "season": ["2022 thru 2023"],
            "start_date": ["2022-01-01"],
            "end_date": ["2022-01-31"],
            "report_date": ["2022-02-01"],
            "type": ["incidence"],
            "value": [-0.1],
        }
    )
    with pytest.raises(
        ValueError,
        match=(
            "^The observations DataFrame contains "
            "negative values in the 'value' column.$"
        ),
    ):
        _validate_and_format_observations(observations)


def test_invalid_types_raises_value_error() -> None:
    """Observations with invalid types in the `value` column raises a `ValueError`."""
    observations = pd.DataFrame(
        data={
            "season": ["2022 thru 2023"],
            "start_date": ["2022-01-01"],
            "end_date": ["2022-01-31"],
            "report_date": ["2022-02-01"],
            "type": ["invalid"],
            "value": [0.0],
        }
    )
    with pytest.raises(
        ValueError,
        match=(
            "^The observations DataFrame contains invalid "
            "values in the 'type' column, must be one of .*.$"
        ),
    ):
        _validate_and_format_observations(observations)


def test_type_other_than_incidence_raises_not_implemented_error() -> None:
    """Observations with non-'incidence' types raises a `NotImplementedError`."""
    observations = pd.DataFrame(
        data={
            "season": ["2024/25"],
            "start_date": ["2025-01-01"],
            "end_date": ["2025-01-31"],
            "report_date": ["2025-01-31"],
            "type": ["prevalence"],
            "value": [0.1],
        }
    )
    with pytest.raises(
        NotImplementedError,
        match=(
            "^Only 'incidence' data is supported, 'prevalence' "
            "and count equivalents are planned.$"
        ),
    ):
        _validate_and_format_observations(observations)


def test_observations_with_report_date_raises_not_implemented_error() -> None:
    """Observations with a 'report_date' raises a `NotImplementedError`."""
    observations = pd.DataFrame(
        data={
            "season": ["2017/18", "2017/18"],
            "start_date": ["2017-11-01", "2017-11-01"],
            "end_date": ["2017-11-07", "2017-11-07"],
            "report_date": ["2017-11-08", "2017-11-09"],
            "type": ["incidence", "incidence"],
            "value": [0.1, 0.11],
        }
    )
    with pytest.raises(
        NotImplementedError,
        match=(
            "^Observations with differing report dates were provided, "
            "nowcasting is not currently supported but planned.$"
        ),
    ):
        _validate_and_format_observations(observations)


@pytest.mark.parametrize(
    "observations",
    [
        pd.DataFrame(
            data={
                "season": ["Winter 2021"],
                "start_date": ["2025-01-01"],
                "end_date": ["2025-01-31"],
                "type": ["incidence"],
                "value": [0.5],
            }
        ),
        pd.DataFrame(
            data={
                "season": ["2020/21", "2020/21"],
                "start_date": ["1/1/2021", "2/1/2021"],
                "end_date": ["1/31/2021", "2/28/2021"],
                "type": ["incidence", "incidence"],
                "value": [0.55, 0.62],
            }
        ),
    ],
)
def test_returns_copy_of_original_data_frame(observations: pd.DataFrame) -> None:
    """The function uses a copy of the original data frame."""
    fmt_observations = _validate_and_format_observations(observations)
    assert fmt_observations is not observations
    assert fmt_observations.index is not observations.index
    assert fmt_observations.to_numpy() is not observations.to_numpy()


@pytest.mark.parametrize(
    "observations",
    [
        pd.DataFrame(
            data={
                "season": ["2019-2020", "2019-2020"],
                "start_date": ["2020-01-01", "2020-01-02"],
                "end_date": ["2020-01-01", "2020-01-02"],
                "report_date": ["2020-01-01", "2020-01-02"],
                "type": ["incidence", "incidence"],
                "value": [0.4, 0.5],
            }
        ),
        pd.DataFrame(
            data={
                "season": ["2020/21", "2020/21"],
                "start_date": ["1/1/2021", "1/8/2021"],
                "end_date": ["1/7/2021", "1/14/2021"],
                "other_date": ["1/8/2021", "1/15/2021"],
                "type": ["incidence", "incidence"],
                "value": [0.123, 0.234],
            }
        ),
        pd.DataFrame(
            data={
                "season": [2021, 2021],
                "start_date": [date(2021, 1, 1), date(2021, 1, 8)],
                "end_date": [date(2021, 1, 7), date(2021, 1, 14)],
                "report_date": [date(2021, 1, 8), date(2021, 1, 15)],
                "type": ["incidence", "incidence"],
                "value": ["0.1", "0.2"],
            }
        ),
    ],
)
def test_columns_coerced_to_expected_types(observations: pd.DataFrame) -> None:
    """The columns of observations are coerced to the expected types."""
    fmt_observations = _validate_and_format_observations(observations)
    assert is_string_dtype(fmt_observations["season"])
    assert all(
        is_datetime64_any_dtype(fmt_observations[col])
        for col in ["start_date", "end_date", "report_date"]
        if col in fmt_observations.columns
    )
    assert isinstance(fmt_observations["type"].dtype, pd.CategoricalDtype)
    assert is_numeric_dtype(fmt_observations["value"])
