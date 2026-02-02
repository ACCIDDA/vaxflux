"""Tools for specifying date ranges for uptake models and scenarios."""

__all__: list[str] = []


from collections.abc import Sequence
from datetime import date, datetime, timedelta
from typing import Final, Literal, NamedTuple, TypeVar

import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from vaxflux._util import _collect_args

_INFER_RANGES_REQUIRED_COLUMNS: Final[dict[Literal["date", "season"], set[str]]] = {
    "date": {"season", "start_date", "end_date", "report_date"},
    "season": {"season", "start_date", "end_date"},
}


class SeasonRange(BaseModel):
    """
    A representation of a season range for uptake scenarios.

    Examples:
        >>> from vaxflux import SeasonRange
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
          Value error, The end date, 2023-12-01, must be after or the same as the start date 2024-03-31. [...]
            For further information visit ...

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
        >>> from vaxflux import DateRange
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
          Value error, The end date, 2023-11-30, must be after or the same as the start date 2023-12-01. [...]
            For further information visit ...
        >>> DateRange(
        ...     season="2023/2024",
        ...     start_date="2023-12-01",
        ...     end_date="2023-12-31",
        ...     report_date="2023-12-30",
        ... )
        Traceback (most recent call last):
            ...
        pydantic_core._pydantic_core.ValidationError: 1 validation error for DateRange
          Value error, The report date, 2023-12-30, must be after or the same as the end date 2023-12-31. [...]
            For further information visit ...

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


def _date_ranges_overlap(date_range1: DateRange, date_range2: DateRange) -> bool:
    """
    Check if two date ranges have overlapping date ranges.

    Args:
        date_range1: First date range to check.
        date_range2: Second date range to check.

    Returns:
        True if the date ranges overlap, False otherwise.

    Examples:
        >>> from datetime import date
        >>> from vaxflux import DateRange
        >>> from vaxflux._dates import _date_ranges_overlap
        >>> dr1 = DateRange(
        ...     season="2023/2024",
        ...     start_date=date(2023, 12, 1),
        ...     end_date=date(2023, 12, 7),
        ...     report_date=date(2023, 12, 8),
        ... )
        >>> dr2 = DateRange(
        ...     season="2023/2024",
        ...     start_date=date(2023, 12, 8),
        ...     end_date=date(2023, 12, 14),
        ...     report_date=date(2023, 12, 15),
        ... )
        >>> _date_ranges_overlap(dr1, dr2)
        False
        >>> # Overlapping date ranges
        >>> dr3 = DateRange(
        ...     season="2023/2024",
        ...     start_date=date(2023, 12, 5),
        ...     end_date=date(2023, 12, 10),
        ...     report_date=date(2023, 12, 11),
        ... )
        >>> _date_ranges_overlap(dr1, dr3)
        True

    """
    return (
        date_range1.start_date <= date_range2.end_date
        and date_range2.start_date <= date_range1.end_date
    )


def _seasons_overlap(season1: SeasonRange, season2: SeasonRange) -> bool:
    """
    Check if two seasons have overlapping date ranges.

    Two seasons overlap if any date appears in both date ranges. This is true when
    the start of one season is on or before the end of the other season, and vice versa.

    Args:
        season1: First season to check.
        season2: Second season to check.

    Returns:
        True if the seasons overlap, False otherwise.

    Examples:
        >>> from datetime import date
        >>> from vaxflux import SeasonRange
        >>> from vaxflux._dates import _seasons_overlap
        >>> season1 = SeasonRange(
        ...     season="2023/2024",
        ...     start_date=date(2023, 12, 1),
        ...     end_date=date(2024, 3, 31),
        ... )
        >>> season2 = SeasonRange(
        ...     season="2024/2025",
        ...     start_date=date(2024, 12, 1),
        ...     end_date=date(2025, 3, 31),
        ... )
        >>> _seasons_overlap(season1, season2)
        False
        >>> # Overlapping seasons
        >>> season3 = SeasonRange(
        ...     season="2024 Winter",
        ...     start_date=date(2024, 3, 1),
        ...     end_date=date(2024, 5, 31),
        ... )
        >>> _seasons_overlap(season1, season3)
        True
        >>> # Adjacent seasons (no overlap)
        >>> season4 = SeasonRange(
        ...     season="2024 Spring",
        ...     start_date=date(2024, 4, 1),
        ...     end_date=date(2024, 6, 30),
        ... )
        >>> _seasons_overlap(season1, season4)
        False
        >>> # Single day overlap
        >>> season5 = SeasonRange(
        ...     season="2024 Q2",
        ...     start_date=date(2024, 3, 31),
        ...     end_date=date(2024, 6, 30),
        ... )
        >>> _seasons_overlap(season1, season5)
        True
        >>> # One season contains another
        >>> season6 = SeasonRange(
        ...     season="January 2024",
        ...     start_date=date(2024, 1, 1),
        ...     end_date=date(2024, 1, 31),
        ... )
        >>> _seasons_overlap(season1, season6)
        True
        >>> # Identical date ranges
        >>> season7 = SeasonRange(
        ...     season="Alt 2023/2024",
        ...     start_date=date(2023, 12, 1),
        ...     end_date=date(2024, 3, 31),
        ... )
        >>> _seasons_overlap(season1, season7)
        True

    """
    return (
        season1.start_date <= season2.end_date
        and season2.start_date <= season1.end_date
    )


Range = TypeVar("Range", bound="DateRange | SeasonRange")


def _collect_ranges(
    args: tuple[Range | Sequence[Range], ...],
    expected_type: type[Range],
    type_name: str,
) -> list[Range]:
    """Collect and validate range objects from arguments.

    Args:
        args: Variable arguments that can be range objects or sequences.
        expected_type: The expected type (SeasonRange or DateRange).
        type_name: Name of the type for error messages.

    Returns:
        List of validated range objects.

    Raises:
        TypeError: If arguments are not of the expected type.
    """
    return _collect_args(args, expected_type, type_name)


def _validate_ranges(
    new_ranges: list[Range],
    existing_ranges: list[Range],
) -> None:
    """Validate ranges for duplicates and overlaps.

    For SeasonRange objects, checks for:
    - Duplicate season names within new ranges
    - Duplicate season names against existing ranges
    - Overlapping date ranges

    For DateRange objects, checks for:
    - Exact duplicate date ranges within new ranges
    - Exact duplicate date ranges against existing ranges
    - Overlapping date ranges

    Args:
        new_ranges: New range objects to validate.
        existing_ranges: Existing range objects to check against.

    Raises:
        ValueError: If duplicate or overlapping ranges are found.
    """
    if not new_ranges:
        return

    # Determine type and apply appropriate validations
    if isinstance(new_ranges[0], SeasonRange):
        _validate_season_ranges(
            new_ranges,  # type: ignore[arg-type]
            existing_ranges,  # type: ignore[arg-type]
        )
    else:  # DateRange
        _validate_date_ranges(
            new_ranges,  # type: ignore[arg-type]
            existing_ranges,  # type: ignore[arg-type]
        )


def _validate_season_ranges(
    new_seasons: list[SeasonRange],
    existing_seasons: list[SeasonRange],
) -> None:
    """Validate season ranges for duplicates and overlaps.

    Args:
        new_seasons: New season ranges to validate.
        existing_seasons: Existing season ranges to check against.

    Raises:
        ValueError: If duplicate season names or overlapping ranges are found.
    """
    # Check for duplicate season names within the new seasons
    new_season_names = [season.season for season in new_seasons]
    if len(new_season_names) != len(set(new_season_names)):
        duplicates = {
            name for name in new_season_names if new_season_names.count(name) > 1
        }
        msg = f"Duplicate season names found in new seasons: {sorted(duplicates)}."
        raise ValueError(msg)

    # Check for duplicate season names against existing seasons
    existing_season_names = {season.season for season in existing_seasons}
    conflicting_names = existing_season_names & set(new_season_names)
    if conflicting_names:
        msg = f"Season names already exist in the model: {sorted(conflicting_names)}."
        raise ValueError(msg)

    # Check for overlapping date ranges within new seasons
    for i, season1 in enumerate(new_seasons):
        for season2 in new_seasons[i + 1 :]:
            if _seasons_overlap(season1, season2):
                msg = (
                    f"Overlapping date ranges found: "
                    f"'{season1.season}' ({season1.start_date} to "
                    f"{season1.end_date}) and '{season2.season}' "
                    f"({season2.start_date} to {season2.end_date})."
                )
                raise ValueError(msg)

    # Check for overlapping date ranges against existing seasons
    for existing_season in existing_seasons:
        for new_season in new_seasons:
            if _seasons_overlap(existing_season, new_season):
                msg = (
                    f"Overlapping date ranges found: "
                    f"'{existing_season.season}' ({existing_season.start_date} "
                    f"to {existing_season.end_date}) and '{new_season.season}' "
                    f"({new_season.start_date} to {new_season.end_date})."
                )
                raise ValueError(msg)


def _validate_date_ranges(  # noqa: C901
    new_dates: list[DateRange],
    existing_dates: list[DateRange],
) -> None:
    """Validate date ranges for exact duplicates and overlaps.

    Args:
        new_dates: New date ranges to validate.
        existing_dates: Existing date ranges to check against.

    Raises:
        ValueError: If duplicate or overlapping date ranges are found.
    """
    # Check for exact duplicates within new dates
    for i, date1 in enumerate(new_dates):
        for date2 in new_dates[i + 1 :]:
            if date1 == date2:
                msg = (
                    f"Duplicate date range found: season='{date1.season}', "
                    f"dates={date1.start_date} to {date1.end_date}, "
                    f"report={date1.report_date}."
                )
                raise ValueError(msg)

    # Check for exact duplicates against existing dates
    for existing_date in existing_dates:
        for new_date in new_dates:
            if existing_date == new_date:
                msg = (
                    f"Date range already exists: season='{new_date.season}', "
                    f"dates={new_date.start_date} to {new_date.end_date}, "
                    f"report={new_date.report_date}."
                )
                raise ValueError(msg)

    # Check for overlapping date ranges within new dates
    for i, date1 in enumerate(new_dates):
        for date2 in new_dates[i + 1 :]:
            if _date_ranges_overlap(date1, date2):
                msg = (
                    "Overlapping date ranges found: "
                    f"'{date1.season}' ({date1.start_date} to {date1.end_date}) "
                    f"and '{date2.season}' ({date2.start_date} to "
                    f"{date2.end_date})."
                )
                raise ValueError(msg)

    # Check for overlapping date ranges against existing dates
    for existing_date in existing_dates:
        for new_date in new_dates:
            if _date_ranges_overlap(existing_date, new_date):
                msg = (
                    f"Overlapping date ranges found: "
                    f"existing '{existing_date.season}' "
                    f"({existing_date.start_date} to {existing_date.end_date}) "
                    f"and new '{new_date.season}' "
                    f"({new_date.start_date} to {new_date.end_date})."
                )
                raise ValueError(msg)


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
