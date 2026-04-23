from typing import TYPE_CHECKING, Any, Final, cast

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from vaxflux._dates import DateRange, SeasonRange

if TYPE_CHECKING:
    from datetime import date

_OBSERVATION_TYPE_CATEGORIES: Final = ("incidence", "prevalence")
_REQUIRED_OBSERVATION_COLUMNS: Final = {
    "season",
    "start_date",
    "end_date",
    "type",
    "value",
}
_METADATA_COLUMNS: Final = {
    "season",
    "start_date",
    "end_date",
    "report_date",
    "type",
    "value",
}


class VaxfluxObservations:
    """
    Container for vaxflux observations data.

    Wraps a validated pandas DataFrame and provides a stable interface for
    working with observations. Use `from_dataframe` to construct an
    instance from a raw DataFrame.

    The underlying DataFrame is accessible via the `data` property.
    Columns that are not part of the standard metadata schema are exposed via
    the `covariate_columns` property for downstream use by the model.

    Examples:
        >>> import pandas as pd
        >>> from vaxflux import VaxfluxObservations
        >>> obs = VaxfluxObservations.from_dataframe(
        ...     pd.DataFrame(
        ...         {
        ...             "season": ["2023/2024"],
        ...             "start_date": ["2023-10-01"],
        ...             "end_date": ["2023-10-31"],
        ...             "type": ["incidence"],
        ...             "value": [0.15],
        ...             "location": ["ABC County"],
        ...         }
        ...     )
        ... )
        >>> obs.data.shape
        (1, 7)
        >>> obs.covariate_columns
        ['location']
        >>> len(obs)
        1
        >>> obs["season"]
        0    2023/2024
        Name: season, dtype: str
    """

    def __init__(self, data: pd.DataFrame) -> None:
        self._data = _validate_and_format_observations(data)

    @classmethod
    def from_dataframe(
        cls,
        data: "VaxfluxObservations | pd.DataFrame",
    ) -> "VaxfluxObservations":
        """
        Construct a `VaxfluxObservations` from a DataFrame or existing instance.

        If `data` is already a `VaxfluxObservations`, it is returned
        unchanged. If it is a `pandas.DataFrame`, it is validated and
        wrapped.

        Args:
            data: A raw observations DataFrame or an existing `VaxfluxObservations`
                instance.

        Returns:
            A validated `VaxfluxObservations` instance.

        Raises:
            ValueError: If the DataFrame is empty or missing required columns.
            ValueError: If the `value` column contains NaN or negative values.
            ValueError: If the `type` column contains unsupported values.
            NotImplementedError: If prevalence observations are provided.
            NotImplementedError: If observations with differing report dates are
                provided (nowcasting is not yet supported).

        """
        if isinstance(data, VaxfluxObservations):
            return data
        return cls(data)

    @classmethod
    def from_csv(cls, *args: Any, **kwargs: Any) -> "VaxfluxObservations":
        """
        Construct a `VaxfluxObservations` by reading a CSV with pandas.

        This is a thin wrapper around `pandas.read_csv`; all positional and keyword
        arguments are forwarded directly to that function before the resulting
        DataFrame is validated and wrapped.

        Args:
            *args: Positional arguments forwarded to `pandas.read_csv`.
            **kwargs: Keyword arguments forwarded to `pandas.read_csv`.

        Returns:
            A validated `VaxfluxObservations` instance.

        Raises:
            Exception: Propagates any exception raised by `pandas.read_csv`.
            ValueError: If the resulting DataFrame is empty or missing required
                columns.
            ValueError: If the `value` column contains NaN or negative values.
            ValueError: If the `type` column contains unsupported values.
            NotImplementedError: If prevalence observations are provided.
            NotImplementedError: If observations with differing report dates are
                provided.
        """
        return cls.from_dataframe(pd.read_csv(*args, **kwargs))

    @property
    def data(self) -> pd.DataFrame:
        """The validated underlying observations DataFrame."""
        return self._data

    @property
    def covariate_columns(self) -> list[str]:
        """
        Column names that are not part of the standard metadata schema.

        Returns:
            A sorted list of column names that are not in the set of known
            metadata columns (`season`, `start_date`, `end_date`,
            `report_date`, `type`, `value`).

        """
        return sorted(col for col in self._data.columns if col not in _METADATA_COLUMNS)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: str) -> pd.Series:
        return self._data[key]

    def to_season_ranges(self) -> list[SeasonRange]:
        """
        Extract season ranges from the observations.

        Returns:
            A list of `SeasonRange` objects, one per observed season, sorted by
            season name.

        Examples:
            >>> import pandas as pd
            >>> from vaxflux import VaxfluxObservations
            >>> observations = VaxfluxObservations.from_dataframe(
            ...     pd.DataFrame(
            ...         {
            ...             "season": ["2023/2024", "2023/2024", "2024/2025"],
            ...             "start_date": ["2023-10-01", "2023-11-01", "2024-10-01"],
            ...             "end_date": ["2023-10-31", "2023-11-30", "2024-10-31"],
            ...             "type": ["incidence", "incidence", "incidence"],
            ...             "value": [0.1, 0.2, 0.3],
            ...         }
            ...     )
            ... )
            >>> season_ranges = observations.to_season_ranges()
            >>> [season_range.season for season_range in season_ranges]
            ['2023/2024', '2024/2025']
            >>> season_ranges[0].start_date, season_ranges[0].end_date
            (datetime.date(2023, 10, 1), datetime.date(2023, 11, 30))
        """
        season_bounds = (
            self._data.groupby("season", as_index=False)
            .agg(
                start_date=("start_date", "min"),
                end_date=("end_date", "max"),
            )
            .sort_values("season")
        )
        seasons = season_bounds["season"].astype(str).tolist()
        start_dates = cast("list[date]", season_bounds["start_date"].dt.date.tolist())
        end_dates = cast("list[date]", season_bounds["end_date"].dt.date.tolist())
        return [
            SeasonRange(
                season=season,
                start_date=start_date,
                end_date=end_date,
            )
            for season, start_date, end_date in zip(
                seasons, start_dates, end_dates, strict=True
            )
        ]

    def to_date_ranges(self) -> list[DateRange]:
        """
        Extract date ranges from the observations.

        Returns:
            A list of distinct `DateRange` objects sorted by season,
            start date, end date, and report date.

        Examples:
            >>> import pandas as pd
            >>> from vaxflux import VaxfluxObservations
            >>> observations = VaxfluxObservations.from_dataframe(
            ...     pd.DataFrame(
            ...         {
            ...             "season": ["2023/2024", "2023/2024"],
            ...             "start_date": ["2023-10-01", "2023-11-01"],
            ...             "end_date": ["2023-10-31", "2023-11-30"],
            ...             "report_date": ["2023-11-01", "2023-12-01"],
            ...             "type": ["incidence", "incidence"],
            ...             "value": [0.1, 0.2],
            ...         }
            ...     )
            ... )
            >>> date_ranges = observations.to_date_ranges()
            >>> len(date_ranges)
            2
            >>> date_ranges[0].start_date, date_ranges[0].report_date
            (datetime.date(2023, 10, 1), datetime.date(2023, 11, 1))
        """
        date_bounds = (
            self._data[["season", "start_date", "end_date", "report_date"]]
            .drop_duplicates(ignore_index=True)
            .sort_values(
                ["season", "start_date", "end_date", "report_date"],
                ignore_index=True,
            )
        )
        seasons = date_bounds["season"].astype(str).tolist()
        start_dates = cast("list[date]", date_bounds["start_date"].dt.date.tolist())
        end_dates = cast("list[date]", date_bounds["end_date"].dt.date.tolist())
        report_dates = cast("list[date]", date_bounds["report_date"].dt.date.tolist())
        return [
            DateRange(
                season=season,
                start_date=start_date,
                end_date=end_date,
                report_date=report_date,
            )
            for season, start_date, end_date, report_date in zip(
                seasons, start_dates, end_dates, report_dates, strict=True
            )
        ]


def _validate_and_format_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and format a raw observations DataFrame.

    Args:
        observations: The observations DataFrame to validate and format.

    Returns:
        A validated and formatted copy of the observations DataFrame.

    Raises:
        ValueError: If the observations DataFrame is empty.
        ValueError: If the observations DataFrame is missing required columns:
            `season`, `start_date`, `end_date`, `type`, `value`.
        ValueError: If the `value` column contains invalid (non-numeric) values.
        ValueError: If the `value` column contains negative values.
        ValueError: If the `type` column contains values other than
            `'incidence'` or `'prevalence'`.
        NotImplementedError: If the observations DataFrame contains only
            `'prevalence'` types (not yet supported).
        NotImplementedError: If the observations DataFrame contains differing
            report dates (nowcasting is not yet supported).

    """
    if not len(observations):
        msg = "No observations provided."
        raise ValueError(msg)
    observation_columns = set(observations.columns)
    if missing_columns := _REQUIRED_OBSERVATION_COLUMNS - observation_columns:
        msg = (
            "The observations DataFrame is missing "
            f"required columns: {missing_columns}."
        )
        raise ValueError(msg)
    observations = observations.copy()
    observations["season"] = observations["season"].astype(str)
    observations["value"] = pd.to_numeric(observations["value"])
    if observations["value"].isna().any():
        msg = (
            "The observations DataFrame contains invalid values in the 'value' column."
        )
        raise ValueError(msg)
    if observations["value"].lt(0).any():
        msg = (
            "The observations DataFrame contains negative values in the 'value' column."
        )
        raise ValueError(msg)
    type_values = observations["type"].astype(str)
    invalid_types = ~type_values.isin(_OBSERVATION_TYPE_CATEGORIES)
    if invalid_types.any():
        msg = (
            "The observations DataFrame contains invalid values in the "
            f"'type' column, must be one of {_OBSERVATION_TYPE_CATEGORIES}."
        )
        raise ValueError(msg)
    observations["type"] = pd.Categorical(
        type_values,
        categories=_OBSERVATION_TYPE_CATEGORIES,
    )
    if {"incidence"} != set(observations["type"].unique().tolist()):
        msg = (
            "Only 'incidence' data is supported, 'prevalence' and count equivalents "
            "are planned."
        )
        raise NotImplementedError(msg)
    for col in {"start_date", "end_date", "report_date"}.intersection(
        observation_columns,
    ):
        if not is_datetime64_any_dtype(observations[col]):
            observations[col] = pd.to_datetime(observations[col])
    if "report_date" not in observations.columns:
        observations["report_date"] = observations["end_date"].copy()
    unique_start_end = observations.drop_duplicates(["start_date", "end_date"])
    unique_start_end_report = observations.drop_duplicates(
        ["start_date", "end_date", "report_date"],
    )
    if len(unique_start_end) != len(unique_start_end_report):
        msg = (
            "Observations with differing report dates were provided, "
            "nowcasting is not currently supported but planned."
        )
        raise NotImplementedError(msg)
    return observations
