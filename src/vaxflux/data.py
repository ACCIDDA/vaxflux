"""
Example datasets and data getting utilities.

This module contains functions for either reading in sample datasets or pulling
data from external data providers.
"""

__all__ = (
    "coordinates_from_incidence",
    "create_logistic_sample_dataset",
    "format_incidence_dataframe",
    "get_ncird_weekly_cumulative_vaccination_coverage",
    "sample_dataset",
)


import io
import time
from datetime import datetime, timezone
from typing import Literal, TypedDict, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import requests
from scipy.special import expit

from vaxflux.covariates import CovariateCategories, _covariate_categories_product
from vaxflux.curves import Curve
from vaxflux.dates import DateRange, SeasonRange


def get_ncird_weekly_cumulative_vaccination_coverage() -> pd.DataFrame:
    """
    Get weekly cumulative vaccination coverage data provided by NCIRD.

    More information about this data can be found on the CDC data page for this
    dataset: `Weekly Cumulative Influenza Vaccination Coverage, Adults 18 and Older, United States <https://data.cdc.gov/Flu-Vaccinations/Weekly-Cumulative-Influenza-Vaccination-Coverage-A/2v3t-r3np/about_data>`_.

    Returns:
        A pandas DataFrame with the columns 'geographic_level', 'geographic_name',
        'demographic_level', 'demographic_name', 'indicator_label',
        'indicator_category_label', 'month_week', 'nd_weekly_estimate',
        'ci_half_width_95pct', 'n_unweighted', 'suppression_flag',
        'current_season_week_ending', 'influenza_season', 'legend',
        'indicator_category_label_sort', 'demographic_level_sort',
        'demographic_name_sort', 'geographic_sort', 'season_sort',
        'legend_sort', '95_ci_lower', and '95_ci_upper'.

    """  # noqa: E501
    now = datetime.now(timezone.utc)
    cache_bust = time.mktime(now.timetuple())
    date = now.strftime("%Y%m%d")
    url = (
        "https://data.cdc.gov/api/views/2v3t-r3np/rows.csv?fourfour=2v3t-r3np"
        f"&cacheBust={cache_bust}&date={date}&accessType=DOWNLOAD"
    )
    # Get and parse the data
    resp = requests.get(url, timeout=30)
    ncird_df = pd.read_csv(
        io.BytesIO(resp.content),
        dtype={
            "Geographic_Level": "string",
            "Geographic_Name": "string",
            "Demographic_Level": "string",
            "Demographic_Name": "string",
            "Indicator_Label": "string",
            "Indicator_Category_Label": "string",
            "Month_Week": "string",
            "Week_Ending": "string",  # empty
            "ND_Weekly_Estimate": "Float64",
            "CI_Half_width_95pct": "Float64",
            "n_unweighted": "UInt64",
            "Suppression_Flag": "boolean",  # 1/0 for True/False
            "Current_Season_Week_Ending": "string",  # datetime string
            "Influenza_Season": "string",
            "Legend": "string",
            "95 CI (%)": "string",  # 2 numbers with a dash
            "Indicator_Category_Label_Sort": "UInt64",
            "Demographic_Level_Sort": "UInt64",
            "Demographic_Name_Sort": "UInt64",
            "Geographic_Sort": "UInt64",
            "Season_Sort": "UInt64",
            "Legend_Sort": "UInt64",
        },
    )
    # Format the data
    ncird_df = ncird_df.rename(
        columns={
            "Geographic_Level": "geographic_level",
            "Geographic_Name": "geographic_name",
            "Demographic_Level": "demographic_level",
            "Demographic_Name": "demographic_name",
            "Indicator_Label": "indicator_label",
            "Indicator_Category_Label": "indicator_category_label",
            "Month_Week": "month_week",
            "Week_Ending": "week_ending",
            "ND_Weekly_Estimate": "nd_weekly_estimate",
            "CI_Half_width_95pct": "ci_half_width_95pct",
            "n_unweighted": "n_unweighted",
            "Suppression_Flag": "suppression_flag",
            "Current_Season_Week_Ending": "current_season_week_ending",
            "Influenza_Season": "influenza_season",
            "Legend": "legend",
            "95 CI (%)": "95_ci",
            "Indicator_Category_Label_Sort": "indicator_category_label_sort",
            "Demographic_Level_Sort": "demographic_level_sort",
            "Demographic_Name_Sort": "demographic_name_sort",
            "Geographic_Sort": "geographic_sort",
            "Season_Sort": "season_sort",
            "Legend_Sort": "legend_sort",
        },
    )
    # Special handling for select columns
    ncird_df["suppression_flag"] = ncird_df["suppression_flag"].astype("boolean")
    ncird_df["current_season_week_ending"] = pd.to_datetime(
        ncird_df["current_season_week_ending"],
        format="%m/%d/%Y %H:%M:%S %p",
    )
    ncird_df[["95_ci_lower", "95_ci_upper"]] = ncird_df["95_ci"].str.split(
        "-",
        n=1,
        expand=True,
    )
    ncird_df["95_ci_lower"] = pd.to_numeric(ncird_df["95_ci_lower"].str.strip()).astype(
        "Float64",
    )
    ncird_df["95_ci_upper"] = pd.to_numeric(ncird_df["95_ci_upper"].str.strip()).astype(
        "Float64",
    )
    return ncird_df.drop(columns=["week_ending", "95_ci"])
    # Return


def format_incidence_dataframe(incidence: pd.DataFrame) -> pd.DataFrame:
    """
    Format an incidence pandas DataFrame.

    Args:
        incidence: A DataFrame with at least the columns 'time' and 'incidence' and
            optionally 'season', 'strata', 'region'.

    Returns:
        A pandas DataFrame with the columns

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     data={
        ...         "time": [1.0, 1.5, 2.0],
        ...         "incidence": [0.01, 0.02, 0.015],
        ...     }
        ... )
        >>> df
           time  incidence
        0   1.0      0.010
        1   1.5      0.020
        2   2.0      0.015
        >>> format_incidence_dataframe(df)
                season       strata       region  time  incidence
        0  All Seasons  All Stratas  All Regions   1.0      0.010
        1  All Seasons  All Stratas  All Regions   1.5      0.020
        2  All Seasons  All Stratas  All Regions   2.0      0.015

    """
    incidence = incidence.copy()
    incidence_columns = set(incidence.columns.tolist())

    if missing_columns := {"time", "incidence"} - incidence_columns:
        msg = (
            "The `incidence` provided is missing required columns: "
            f"""'{"', '".join(missing_columns)}'."""
        )
        raise ValueError(
            msg,
        )

    for column in ("time", "incidence"):
        incidence[column] = pd.to_numeric(incidence[column]).astype("float64")

    for column in ("season", "strata", "region"):
        if column not in incidence_columns:
            incidence[column] = pd.Series(
                data=len(incidence) * [f"All {column.capitalize()}s"],
                dtype="string",
            )
        else:
            incidence[column] = incidence[column].astype("string")

    return incidence[["season", "strata", "region", "time", "incidence"]]


def coordinates_from_incidence(
    incidence: pd.DataFrame,
) -> dict[Literal["season", "region", "strata", "observation"], list[str]]:
    """
    Extract model coordinates from an incidence pandas DataFrame.

    Args:
        incidence: A formatted incidence pandas DataFrame.

    Returns:
        A dictionary of coordinates that can be provided to xarray or PyMC.

    """
    keys: tuple[Literal["season", "region", "strata"], ...] = (
        "season",
        "region",
        "strata",
    )
    coords: dict[Literal["season", "region", "strata", "observation"], list[str]] = {
        **{v: np.sort(incidence[v].unique()).tolist() for v in keys},
        "observation": np.arange(len(incidence)).astype(str).tolist(),
    }
    return coords


class ParametersRow(TypedDict):
    season: str
    strata: str
    region: str
    m: float
    r: float
    s: float


def create_logistic_sample_dataset(
    parameters: pd.DataFrame,
    time: npt.NDArray[np.float64],
    epsilon: float,
    error: Literal["gamma", "normal"] | None = "gamma",
    seed: int = 0,
) -> pd.DataFrame:
    """
    Create a synthetic logistic incidence dataset.

    Args:
        parameters: A pandas DataFrame with the columns 'season', 'strata', 'region',
            'm', 'r', and 's'.
        time: A numpy array of the time steps to generate a dataset for.
        epsilon: The standard deviation to use in the resulting observations.
        error: The error distribution to use in generating the observed incidences or
            `None` for no noise added to the dataset.
        seed: An integer corresponding to the random seed to use when generating a
            dataset for consistency across calls.

    Returns:
        A formatted incidence dataset.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from vaxflux.data import create_logistic_sample_dataset
        >>> parameters = pd.DataFrame(
        ...     data={
        ...         "season": ["2023/24"],
        ...         "strata": ["All stratas"],
        ...         "region": ["All regions"],
        ...         "m": [0.5],
        ...         "r": [0.3],
        ...         "s": [20.0],
        ...     },
        ... )
        >>> parameters
            season       strata       region    m    r     s
        0  2023/24  All stratas  All regions  0.5  0.3  20.0
        >>> time = np.arange(40, step=3)
        >>> create_logistic_sample_dataset(parameters, time, 0.001)
             season       strata       region  time     value
        0   2023/24  All stratas  All regions   0.0  0.000128
        1   2023/24  All stratas  All regions   3.0  0.001984
        2   2023/24  All stratas  All regions   6.0  0.005459
        3   2023/24  All stratas  All regions   9.0  0.007348
        4   2023/24  All stratas  All regions  12.0  0.014066
        5   2023/24  All stratas  All regions  15.0  0.027984
        6   2023/24  All stratas  All regions  18.0  0.044186
        7   2023/24  All stratas  All regions  21.0  0.046088
        8   2023/24  All stratas  All regions  24.0  0.033544
        9   2023/24  All stratas  All regions  27.0  0.019666
        10  2023/24  All stratas  All regions  30.0  0.008194
        11  2023/24  All stratas  All regions  33.0  0.001570
        12  2023/24  All stratas  All regions  36.0  0.001995
        13  2023/24  All stratas  All regions  39.0  0.000210

    """
    rs = np.random.RandomState(seed)
    incidence = []
    for row in cast("list[ParametersRow]", parameters.to_dict(orient="records")):
        tmp = np.exp(-row["r"] * (time - row["s"]))
        mu = expit(row["m"]) * row["r"] * tmp * np.power(1.0 + tmp, -2.0)
        if error == "gamma":
            obs = rs.gamma(shape=np.power(mu / epsilon, 2.0), scale=(epsilon**2.0) / mu)
        elif error == "normal":
            obs = np.maximum(rs.normal(loc=mu, scale=epsilon), 0.0)
        else:
            obs = mu
        incidence.append(
            pd.DataFrame(
                data={
                    "season": len(time) * [row["season"]],
                    "strata": len(time) * [row["strata"]],
                    "region": len(time) * [row["region"]],
                    "time": time,
                    "incidence": obs,
                },
            ),
        )
    incidence_df = pd.concat(incidence, ignore_index=True)
    incidence_df = format_incidence_dataframe(incidence_df)
    return incidence_df.rename(columns={"incidence": "value"})


def sample_dataset(
    curve: Curve,
    season_ranges: list[SeasonRange],
    date_ranges: list[DateRange],
    covariate_categories: list[CovariateCategories],
    parameters: list[tuple[str | float, ...]],
    epsilon: float,
    random_seed: int = 1,
) -> pd.DataFrame:
    """
    Generate a sample dataset from the given incidence curve.

    Args:
        curve: The incidence curve to sample from.
        season_ranges: The season ranges to sample from.
        date_ranges: The date ranges to generate observations for.
        covariate_categories: The covariate categories to sample from.
        parameters: The parameters to sample from. List of tuples with the first element
            being the curve parameter name, the second element being the season, and the
            following being the covariate categories and the last element being the
            value.
        epsilon: The standard deviation to use in the resulting observations.
        random_seed: The random seed to use for reproducibility.

    Returns:
        A pandas DataFrame of observations with the columns 'season',
        'season_start_date', 'season_end_date', 'start_date', 'end_date', 'report_date',
        'type', and 'value' as well as the covariate categories covariate names.

    """
    generator = np.random.default_rng(seed=random_seed)
    season_ranges_map = {
        season_range.season: season_range for season_range in season_ranges
    }
    categories_prod = _covariate_categories_product(covariate_categories)
    records = []
    for date_range in date_ranges:
        for category_prod in categories_prod:
            season_range = season_ranges_map[date_range.season]
            kwargs = {}
            for parameter in parameters:
                if list(parameter)[1:-1] == [
                    season_range.season,
                    *category_prod.values(),
                ]:
                    kwargs[str(parameter[0])] = np.array(float(parameter[-1]))
            tstart = (date_range.start_date - season_range.start_date).days
            tend = (date_range.end_date - season_range.start_date).days + 1.0
            t0 = np.array([float(i) for i in range(int(tstart), int(tend))])
            t1 = t0 + 1.0
            y = curve.prevalence_difference(t0, t1, **kwargs).eval()
            if epsilon > 0:
                y = generator.gamma(
                    shape=np.power(y / epsilon, 2.0),
                    scale=(epsilon**2.0) / y,
                )
            value = sum(y)
            record = (
                {
                    "season": date_range.season,
                    "season_start_date": season_range.start_date,
                    "season_end_date": season_range.end_date,
                    "start_date": date_range.start_date,
                    "end_date": date_range.end_date,
                    "report_date": date_range.report_date,
                }
                | category_prod
                | {"type": "incidence", "value": value}
            )
            records.append(record)
    observations = pd.DataFrame.from_records(records)
    for col in ("season", "type"):
        observations[col] = observations[col].astype("string")
    for col in (
        "season_start_date",
        "season_end_date",
        "start_date",
        "end_date",
        "report_date",
    ):
        observations[col] = pd.to_datetime(observations[col])
    return observations
