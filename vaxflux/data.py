"""
Example datasets and data getting utilities.

This module contains functions for either reading in sample datasets or pulling
data from external data providers. Current exported functionality includes:
- `read_flu_vacc_data`
- `get_ncird_weekly_cumulative_vaccination_coverage`
"""

__all__ = ["read_flu_vacc_data", "get_ncird_weekly_cumulative_vaccination_coverage"]


from datetime import datetime
from importlib import resources
import io
import time

import numpy as np
import pandas as pd
import requests

from . import sample_data


def read_flu_vacc_data():
    """
    Read a sample dataset of flu vaccination data from 2018 to 2024.

    Returns:
        A structured arrays with five fields named: "season", "end_date", "week",
        "sort_order", and "doses".

    Examples:
        >>> data = read_flu_vacc_data()
        >>> data[:5]
        array([(b'2018-2019', b'8/4/18', 30, 1,  0.52),
            (b'2018-2019', b'8/11/18', 31, 2,  3.23),
            (b'2018-2019', b'8/18/18', 32, 3, 10.18),
            (b'2018-2019', b'8/25/18', 33, 4, 19.99),
            (b'2018-2019', b'9/1/18', 34, 5, 37.38)],
            dtype=[('season', 'S9'), ('end_date', 'S8'), ('week', '<u8'), ('sort_order', '<u8'), ('doses', '<f8')])
        >>> data["doses"][:5]
        array([ 0.52,  3.23, 10.18, 19.99, 37.38])
    """
    csv_file = resources.files(sample_data) / "Flu_Vacc_Data.csv"
    with csv_file.open() as f:
        data = np.genfromtxt(
            f,
            delimiter=",",
            dtype=[
                ("season", "S9"),
                ("end_date", "S8"),
                ("week", "u8"),
                ("sort_order", "u8"),
                ("doses", "f8"),
            ],
            skip_header=1,
        )
    return data


def get_ncird_weekly_cumulative_vaccination_coverage() -> pd.DataFrame:
    """
    Get weekly cumulative vaccination coverage data provided by NCIRD.

    More information about this data can be found on the data.cdc.gov page for this
    dataset: https://tinyurl.com/yc33txdp.

    Returns:
        A pandas DataFrame with the columns 'geographic_level', 'geographic_name',
        'demographic_level', 'demographic_name', 'indicator_label',
        'indicator_category_label', 'month_week', 'nd_weekly_estimate',
        'ci_half_width_95pct', 'n_unweighted', 'suppression_flag',
        'current_season_week_ending', 'influenza_season', 'legend',
        'indicator_category_label_sort', 'demographic_level_sort',
        'demographic_name_sort', 'geographic_sort', 'season_sort',
        'legend_sort', '95_ci_lower', and '95_ci_upper'.

    """
    # TODO: This should use the API provided by data.cdc.gov instead of direct download
    # Format the correct url
    now = datetime.now()
    cache_bust = time.mktime(now.timetuple())
    date = now.strftime("%Y%m%d")
    url = (
        "https://data.cdc.gov/api/views/2v3t-r3np/rows.csv?fourfour=2v3t-r3np"
        f"&cacheBust={cache_bust}&date={date}&accessType=DOWNLOAD"
    )
    # Get and parse the data
    resp = requests.get(url)
    df = pd.read_csv(
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
    df = df.rename(
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
        }
    )
    # Special handling for select columns
    df["suppression_flag"] = df["suppression_flag"].astype("boolean")
    df["current_season_week_ending"] = pd.to_datetime(
        df["current_season_week_ending"], format="%m/%d/%Y %H:%M:%S %p"
    )
    df[["95_ci_lower", "95_ci_upper"]] = df["95_ci"].str.split("-", n=1, expand=True)
    df["95_ci_lower"] = pd.to_numeric(df["95_ci_lower"].str.strip()).astype("Float64")
    df["95_ci_upper"] = pd.to_numeric(df["95_ci_upper"].str.strip()).astype("Float64")
    df = df.drop(columns=["week_ending", "95_ci"])
    # Return
    return df
