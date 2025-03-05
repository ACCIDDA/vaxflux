"""Tools for specifying date ranges for uptake models and scenarios."""

__all__ = ("DateRange", "SeasonRange")


from datetime import date, datetime
from typing import Literal, NamedTuple, TypeVar, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict


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
    if not observations and not ranges:
        raise ValueError("At least one of `observations` or `ranges` is required.")
    cls = DateRange if mode == "date" else SeasonRange
    columns = {
        "date": {"season", "start_date", "end_date", "report_date"},
        "season": {"season", "season_start_date", "season_end_date"},
    }[mode]
    if observations:
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
            return [
                cls(row._asdict())  # type: ignore
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
                season_names = {season.season for season in non_explicit_season_ranges}
                raise ValueError(
                    "The observed season ranges are not consistent with the "
                    f"explicit season ranges, not accounting for: {season_names}."
                )
    # Only ranges
    return ranges
