"""Unit tests for `VaxfluxObservations.from_dataframe`."""

from datetime import date
from itertools import chain, combinations
from typing import Final

import pandas as pd
import pytest
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype, is_string_dtype

from vaxflux._vaxflux_observations import (
    _REQUIRED_OBSERVATION_COLUMNS,
    VaxfluxObservations,
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


@pytest.mark.parametrize(
    "observations",
    [
        pd.DataFrame(),
        pd.DataFrame(data={"incidence": []}),
    ],
)
def test_zero_length_data_frame_value_error(observations: pd.DataFrame) -> None:
    """Providing a zero-length DataFrame raises a `ValueError`."""
    with pytest.raises(ValueError, match=r"^No observations provided.$"):
        VaxfluxObservations.from_dataframe(observations)


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
        match=r"^The observations DataFrame is missing required columns: .*.$",
    ):
        VaxfluxObservations.from_dataframe(observations)


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
        },
    )
    with pytest.raises(
        ValueError,
        match=(
            r"^The observations DataFrame contains "
            r"invalid values in the 'value' column.$"
        ),
    ):
        VaxfluxObservations.from_dataframe(observations)


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
        },
    )
    with pytest.raises(
        ValueError,
        match=(
            r"^The observations DataFrame contains "
            r"negative values in the 'value' column.$"
        ),
    ):
        VaxfluxObservations.from_dataframe(observations)


def test_invalid_types_raises_value_error() -> None:
    """Observations with invalid types in the `type` column raises a `ValueError`."""
    observations = pd.DataFrame(
        data={
            "season": ["2022 thru 2023"],
            "start_date": ["2022-01-01"],
            "end_date": ["2022-01-31"],
            "report_date": ["2022-02-01"],
            "type": ["invalid"],
            "value": [0.0],
        },
    )
    with pytest.raises(
        ValueError,
        match=(
            r"^The observations DataFrame contains invalid "
            r"values in the 'type' column, must be one of .*.$"
        ),
    ):
        VaxfluxObservations.from_dataframe(observations)


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
        },
    )
    with pytest.raises(
        NotImplementedError,
        match=(
            r"^Only 'incidence' data is supported, 'prevalence' "
            r"and count equivalents are planned.$"
        ),
    ):
        VaxfluxObservations.from_dataframe(observations)


def test_observations_with_differing_report_dates_raises_not_implemented_error() -> (
    None
):
    """Observations with differing report dates raises a `NotImplementedError`."""
    observations = pd.DataFrame(
        data={
            "season": ["2017/18", "2017/18"],
            "start_date": ["2017-11-01", "2017-11-01"],
            "end_date": ["2017-11-07", "2017-11-07"],
            "report_date": ["2017-11-08", "2017-11-09"],
            "type": ["incidence", "incidence"],
            "value": [0.1, 0.11],
        },
    )
    with pytest.raises(
        NotImplementedError,
        match=(
            r"^Observations with differing report dates were provided, "
            r"nowcasting is not currently supported but planned.$"
        ),
    ):
        VaxfluxObservations.from_dataframe(observations)


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
            },
        ),
        pd.DataFrame(
            data={
                "season": ["2020/21", "2020/21"],
                "start_date": ["1/1/2021", "2/1/2021"],
                "end_date": ["1/31/2021", "2/28/2021"],
                "type": ["incidence", "incidence"],
                "value": [0.55, 0.62],
            },
        ),
    ],
)
def test_data_property_is_copy_of_original_data_frame(
    observations: pd.DataFrame,
) -> None:
    """The underlying data is a copy of the input DataFrame, not the same object."""
    obs = VaxfluxObservations.from_dataframe(observations)
    assert obs.data is not observations
    assert obs.data.index is not observations.index
    assert obs.data.to_numpy() is not observations.to_numpy()


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
            },
        ),
        pd.DataFrame(
            data={
                "season": ["2020/21", "2020/21"],
                "start_date": ["1/1/2021", "1/8/2021"],
                "end_date": ["1/7/2021", "1/14/2021"],
                "other_date": ["1/8/2021", "1/15/2021"],
                "type": ["incidence", "incidence"],
                "value": [0.123, 0.234],
            },
        ),
        pd.DataFrame(
            data={
                "season": [2021, 2021],
                "start_date": [date(2021, 1, 1), date(2021, 1, 8)],
                "end_date": [date(2021, 1, 7), date(2021, 1, 14)],
                "report_date": [date(2021, 1, 8), date(2021, 1, 15)],
                "type": ["incidence", "incidence"],
                "value": ["0.1", "0.2"],
            },
        ),
    ],
)
def test_columns_coerced_to_expected_types(observations: pd.DataFrame) -> None:
    """Columns of the underlying data are coerced to the expected types."""
    obs = VaxfluxObservations.from_dataframe(observations)
    assert is_string_dtype(obs.data["season"])
    assert all(
        is_datetime64_any_dtype(obs.data[col])
        for col in ["start_date", "end_date", "report_date"]
        if col in obs.data.columns
    )
    assert isinstance(obs.data["type"].dtype, pd.CategoricalDtype)
    assert is_numeric_dtype(obs.data["value"])


def test_from_dataframe_passthrough_for_existing_instance() -> None:
    """Providing `VaxfluxObservations` to `from_dataframe` returns it as is."""
    obs = VaxfluxObservations.from_dataframe(PERFECT_OBSERVATIONS)
    assert VaxfluxObservations.from_dataframe(obs) is obs


def test_len_matches_underlying_dataframe() -> None:
    """`len()` returns the number of rows in the underlying DataFrame."""
    obs = VaxfluxObservations.from_dataframe(PERFECT_OBSERVATIONS)
    assert len(obs) == len(PERFECT_OBSERVATIONS)


def test_getitem_delegates_to_underlying_dataframe() -> None:
    """`__getitem__` delegates column access to the underlying DataFrame."""
    obs = VaxfluxObservations.from_dataframe(PERFECT_OBSERVATIONS)
    pd.testing.assert_series_equal(obs["season"], obs.data["season"])


def test_covariate_columns_empty_when_no_extra_columns() -> None:
    """`covariate_columns` is empty when only metadata columns are present."""
    obs = VaxfluxObservations.from_dataframe(PERFECT_OBSERVATIONS)
    assert obs.covariate_columns == []


def test_covariate_columns_returns_extra_columns() -> None:
    """`covariate_columns` returns non-metadata columns in sorted order."""
    observations = pd.DataFrame(
        data={
            "season": ["2020/21", "2020/21"],
            "start_date": ["2021-01-01", "2021-02-01"],
            "end_date": ["2021-01-31", "2021-02-28"],
            "type": ["incidence", "incidence"],
            "value": [0.1, 0.2],
            "age_group": ["18-49", "50+"],
            "region": ["northeast", "northeast"],
        },
    )
    obs = VaxfluxObservations.from_dataframe(observations)
    assert obs.covariate_columns == ["age_group", "region"]
