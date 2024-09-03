"""
Example datasets and data getting utilities.

This module contains functions for either reading in sample datasets or pulling
data from external data providers. Current exported functionality includes:
- `read_flu_vacc_data`
"""

__all__ = ["read_flu_vacc_data"]


from importlib import resources

import numpy as np

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
