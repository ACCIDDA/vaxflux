"""Tools for specifying date ranges for uptake models and scenarios."""

__all__ = ("DateRange", "SeasonRange", "daily_date_ranges")


from datetime import date, datetime, timedelta
from typing import Final, Literal, NamedTuple, TypeVar, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict

_INFER_RANGES_REQUIRED_COLUMNS: Final[dict[Literal["date", "season"], set[str]]] = {
    "date": {"season", "start_date", "end_date", "report_date"},
    "season": {"season", "start_date", "end_date"},
}


class SeasonRange(BaseModel):
    """
    A representation of a season range for uptake scenarios.

    Attributes:
        season: The name of the season for the season range.
        start_date: The start date of the season range, used to make seasonal dates
            relative.
        end_date: The end date of the season range.

    """

    model_config = ConfigDict(frozen=True)

    season: str
    start_date: date
    end_date: date


class DateRange(BaseModel):
    """
    A representation of a date range for uptake scenarios.

    Attributes:
        season: The season for the date range.
        start_date: The start date of the date range.
        end_date: The end date of the date range.
        report_date: The report date of the date range.

    """

    model_config = ConfigDict(frozen=True)

    season: str
    start_date: date
    end_date: date
    report_date: date


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
        raise ValueError(
            "The number of days for each daily date range must be at least 0."
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
                    raise ValueError(
                        "The number of days for each daily date range does not divide "
                        f"evenly into the season range for {season_range.season}."
                    )
                elif remainder == "fill":
                    end_date = season_range.end_date
                else:
                    break
            date_ranges.append(
                DateRange(
                    season=season_range.season,
                    start_date=start_date,
                    end_date=end_date,
                    report_date=end_date,
                )
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
        raise ValueError("At least one of `observations` or `ranges` is required.")
    cls = DateRange if mode == "date" else SeasonRange
    columns = _INFER_RANGES_REQUIRED_COLUMNS[mode]
    if observations is not None:
        # Only observations
        if not ranges:
            if missing_columns := columns - set(observations.columns):
                raise ValueError(
                    f"Missing required columns in the observations: {missing_columns}."
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
                    index=False, name=f"Observation{mode.capitalize()}Row"
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
                non_explicit_season_ranges = cast(set[SeasonRange], non_explicit_ranges)
                season_names = ", ".join(
                    sorted({season.season for season in non_explicit_season_ranges})
                )
                raise ValueError(
                    "The observed season ranges are not consistent with the "
                    f"explicit season ranges, not accounting for: {season_names}."
                )
    # Only ranges
    return ranges
