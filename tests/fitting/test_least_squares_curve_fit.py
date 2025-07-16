"""Unit tests for `vaxflux.fitting.least_squares_curve_fit`."""

import pandas as pd
import pytest

from vaxflux.curves import LogisticCurve
from vaxflux.fitting import least_squares_curve_fit


def test_only_prevalence_type_observations_are_supported_currently() -> None:
    """Only 'prevalence' data is supported currently."""
    with pytest.raises(
        NotImplementedError,
        match=(
            r"^Only 'prevalence' data is supported, 'incidence' and count equivalents "
            r"are planned.$"
        ),
    ):
        least_squares_curve_fit(
            LogisticCurve(),
            observations=pd.DataFrame.from_records(
                [
                    {
                        "season": "2022/23",
                        "start_date": "2022-10-01",
                        "end_date": "2023-10-31",
                        "report_date": "2023-11-01",
                        "type": "incidence",
                        "value": 0.1,
                    }
                ]
            ),
        )


def test_no_covariate_categories_raises_not_implemented_error() -> None:
    """No covariate categories raises a `NotImplementedError`."""
    with pytest.raises(
        NotImplementedError,
        match=(r"^No covariate categories provided, at least one is required\.$"),
    ):
        least_squares_curve_fit(
            LogisticCurve(),
            observations=pd.DataFrame.from_records(
                [
                    {
                        "season": "2022/23",
                        "start_date": "2022-10-01",
                        "end_date": "2023-10-31",
                        "report_date": "2023-11-01",
                        "type": "prevalence",
                        "value": 0.1,
                    }
                ]
            ),
        )
