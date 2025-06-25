"""Tools for specifying date ranges for uptake models and scenarios."""

__all__ = ("DateRange", "SeasonRange", "daily_date_ranges")


from datetime import date, datetime, timedelta
from typing import Final, Literal, NamedTuple, TypeVar

import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

_INFER_RANGES_REQUIRED_COLUMNS: Final[dict[Literal["date", "season"], set[str]]] = {
    "date": {"season", "start_date", "end_date", "report_date"},
    "season": {"season", "start_date", "end_date"},
}


class SeasonRange(BaseModel):
    """
    A representation of a season range for uptake scenarios.

    Examples:
        >>> from vaxflux.dates import SeasonRange
        >>> season_range = SeasonRange(
        ...     season="2023/2024",
        ...     start_date="2023-12-01",
        ...     end_date="2024-03-31",
        ... )
        >>> season_range.season
        '2023/2024'
        >>> season_range.start_date
        datetime.date(2023, 12, 1)
        >>> season_range.end_date
        datetime.date(2024, 3, 31)
        >>> SeasonRange(
        ...     season="2023/2024",
        ...     start_date="2024-03-31",
        ...     end_date="2023-12-01",
        ... )
        Traceback (most recent call last):
            ...
        pydantic_core._pydantic_core.ValidationError: 1 validation error for SeasonRange
          Value error, The end date, 2023-12-01, must be after or the same as the start date 2024-03-31. [type=value_error, input_value={'season': '2023/2024', '...nd_date': '2023-12-01'}, input_type=dict]
            For further information visit https://errors.pydantic.dev/2.11/v/value_error

    """  # noqa: E501

    model_config = ConfigDict(frozen=True)

    #: The name of the season for the season range.
    season: str

    #: The start date of the season range, used to make seasonal dates relative.
    start_date: date

    #: The end date of the season range.
    end_date: date

    @model_validator(mode="after")
    def _validate_date_order(self) -> "SeasonRange":
        """
        Validate the order of the dates in the DateRange.

        Returns:
            The validated DateRange instance.

        Raises:
            ValueError: If the end date is before the start date.
            ValueError: If the report date is before the end date.
        """
        if self.end_date < self.start_date:
            msg = (
                f"The end date, {self.end_date}, must be after "
                f"or the same as the start date {self.start_date}."
            )
            raise ValueError(
                msg,
            )
        return self


class DateRange(BaseModel):
    """
    A representation of a date range for uptake scenarios.

    Examples:
        >>> from vaxflux.dates import DateRange
        >>> date_range = DateRange(
        ...     season="2023/2024",
        ...     start_date="2023-12-01",
        ...     end_date="2023-12-31",
        ...     report_date="2024-01-01",
        ... )
        >>> date_range.season
        '2023/2024'
        >>> date_range.start_date
        datetime.date(2023, 12, 1)
        >>> date_range.end_date
        datetime.date(2023, 12, 31)
        >>> date_range.report_date
        datetime.date(2024, 1, 1)
        >>> DateRange(
        ...     season="2023/2024",
        ...     start_date="2023-12-01",
        ...     end_date="2023-11-30",
        ...     report_date="2023-12-01",
        ... )
        Traceback (most recent call last):
            ...
        pydantic_core._pydantic_core.ValidationError: 1 validation error for DateRange
          Value error, The end date, 2023-11-30, must be after or the same as the start date 2023-12-01. [type=value_error, input_value={'season': '2023/2024', '...ort_date': '2023-12-01'}, input_type=dict]
            For further information visit https://errors.pydantic.dev/2.11/v/value_error
        >>> DateRange(
        ...     season="2023/2024",
        ...     start_date="2023-12-01",
        ...     end_date="2023-12-31",
        ...     report_date="2023-12-30",
        ... )
        Traceback (most recent call last):
            ...
        pydantic_core._pydantic_core.ValidationError: 1 validation error for DateRange
          Value error, The report date, 2023-12-30, must be after or the same as the end date 2023-12-31. [type=value_error, input_value={'season': '2023/2024', '...ort_date': '2023-12-30'}, input_type=dict]
            For further information visit https://errors.pydantic.dev/2.11/v/value_error

    """  # noqa: E501

    model_config = ConfigDict(frozen=True)

    #: The season for the date range.
    season: str

    #: The start date of the date range.
    start_date: date

    #: The end date of the date range.
    end_date: date

    #: The report date of the date range.
    report_date: date

    @model_validator(mode="after")
    def _validate_date_order(self) -> "DateRange":
        """
        Validate the order of the dates in the DateRange.

        Returns:
            The validated DateRange instance.

        Raises:
            ValueError: If the end date is before the start date.
            ValueError: If the report date is before the end date.
        """
        if self.end_date < self.start_date:
            msg = (
                f"The end date, {self.end_date}, must be after "
                f"or the same as the start date {self.start_date}."
            )
            raise ValueError(
                msg,
            )
        if self.report_date < self.end_date:
            msg = (
                f"The report date, {self.report_date}, must be after "
                f"or the same as the end date {self.end_date}."
            )
            raise ValueError(
                msg,
            )
        return self


def daily_date_ranges(
    season_ranges: list[SeasonRange] | SeasonRange,
    range_days: int = 0,
    remainder: Literal["fill", "raise", "skip"] = "raise",
) -> list[DateRange]:
    """
    Create daily date ranges from the season ranges.

    Args:
        season_ranges: The season ranges to create the daily date ranges from.
        range_days: The number of days for each daily date range, must be at least 0.
        remainder: The strategy to handle the remainder of days when the season ranges
            do not divide evenly into daily date ranges. Options are "fill" to fill the
            remainder with the last date range, "raise" to raise an error, and "skip" to
            skip the remainder.

    Returns:
        The daily date ranges for the uptake scenarios.

    Raises:
        ValueError: If the number of days for each daily date range is less than 1.
        ValueError: If the number of days for each daily date range does not divide
            evenly into the season range and `remainder` is 'raise'.

    """
    season_ranges = (
        [season_ranges] if isinstance(season_ranges, SeasonRange) else season_ranges
    )
    if range_days < 0:
        msg = "The number of days for each daily date range must be at least 0."
        raise ValueError(
            msg,
        )
    date_ranges = []
    td = timedelta(days=range_days)
    td_one_day = timedelta(days=1)
    for season_range in season_ranges:
        start_date = season_range.start_date
        while start_date <= season_range.end_date:
            end_date = start_date + td
            if end_date > season_range.end_date:
                if remainder == "raise":
                    msg = (
                        "The number of days for each daily date range does not divide "
                        f"evenly into the season range for {season_range.season}."
                    )
                    raise ValueError(
                        msg,
                    )
                if remainder == "fill":
                    end_date = season_range.end_date
                else:
                    break
            date_ranges.append(
                DateRange(
                    season=season_range.season,
                    start_date=start_date,
                    end_date=end_date,
                    report_date=end_date,
                ),
            )
            start_date = end_date + td_one_day
    return date_ranges


DateOrSeasonRange = TypeVar("DateOrSeasonRange", bound=DateRange | SeasonRange)


class ObservationDateRow(NamedTuple):
    season: str
    start_date: datetime
    end_date: datetime
    report_date: datetime


class ObservationSeasonRow(NamedTuple):
    season: str
    season_start_date: datetime
    season_end_date: datetime


def _infer_ranges_from_observations(
    observations: pd.DataFrame | None,
    ranges: list[DateOrSeasonRange],
    mode: Literal["date", "season"],
) -> list[DateOrSeasonRange]:
    """
    Infer the date or season ranges from the observations.

    Args:
        observations: The uptake dataset to use.
        ranges: The date or season ranges for the uptake scenarios.
        mode: The mode of the inference, either "date" or "season".

    Returns:
        The inferred date or season ranges, depending on the `mode`.

    Raises:
        ValueError: If both `observations` and `ranges` are empty.
        ValueError: If the required columns are missing in the observations.
        ValueError: If the observed season ranges are not consistent with the explicit
            season ranges, only applicable for the season mode.
    """
    if observations is None and not ranges:
        msg = "At least one of `observations` or `ranges` is required."
        raise ValueError(msg)
    cls = DateRange if mode == "date" else SeasonRange
    columns = _INFER_RANGES_REQUIRED_COLUMNS[mode]
    if observations is not None:
        # Only observations
        if not ranges:
            if missing_columns := columns - set(observations.columns):
                msg = (
                    f"Missing required columns in the observations: {missing_columns}."
                )
                raise ValueError(
                    msg,
                )
            observations_ranges = (
                observations[list(columns)]
                .drop_duplicates(ignore_index=True)
                .sort_values(list(columns), ignore_index=True)
            )
            if mode == "season":
                observations_ranges = (
                    observations_ranges.groupby("season")
                    .agg({"start_date": "min", "end_date": "max"})
                    .reset_index()
                )
            return [
                cls.model_validate(row._asdict())  # type: ignore[misc,operator]
                for row in observations_ranges.itertuples(
                    index=False,
                    name=f"Observation{mode.capitalize()}Row",
                )
            ]
        # Both ranges and observations
        if columns.issubset(observations.columns):
            observation_ranges: list[DateOrSeasonRange] = (
                _infer_ranges_from_observations(observations, [], mode)
            )
            if mode == "date":
                return list(set(ranges) | set(observation_ranges))
            if non_explicit_ranges := set(observation_ranges) - set(ranges):
                ranges_map = {r.season: r for r in ranges}
                season_names = ", ".join(
                    sorted(
                        [
                            non_explicit_range.season
                            for non_explicit_range in non_explicit_ranges
                            if (
                                explicit_range := ranges_map.get(
                                    non_explicit_range.season
                                )
                            )
                            is None
                            or (
                                explicit_range.start_date
                                > non_explicit_range.start_date
                                or explicit_range.end_date < non_explicit_range.end_date
                            )
                        ]
                    )
                )
                if season_names:
                    msg = (
                        "The observed season ranges are not consistent with the "
                        f"explicit season ranges: {season_names}. Either they are not "
                        "present in the explicit ranges or the observed dates are not "
                        "within the explicit ranges."
                    )
                    raise ValueError(msg)
    # Only ranges
    return ranges
