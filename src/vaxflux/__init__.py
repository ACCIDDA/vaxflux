"""Model seasonal vaccination uptake curves."""

__all__ = (
    "Covariate",
    "CovariateCategories",
    "Curve",
    "DateRange",
    "Implementation",
    "Intervention",
    "LogisticCurve",
    "PartiallyPooledGaussianCovariate",
    "PooledGaussianCovariate",
    "SeasonRange",
    "SeasonVaryingPartiallyPooledGaussianCovariate",
    "VaxfluxInferenceData",
    "VaxfluxModel",
    "VaxfluxObservations",
    "daily_date_ranges",
    "data",
)
__version__ = "0.4.0dev"


from vaxflux import (
    data,
)
from vaxflux._covariate_categories import CovariateCategories
from vaxflux._covariates import (
    Covariate,
    PartiallyPooledGaussianCovariate,
    PooledGaussianCovariate,
    SeasonVaryingPartiallyPooledGaussianCovariate,
)
from vaxflux._curves import Curve, LogisticCurve
from vaxflux._dates import DateRange, SeasonRange, daily_date_ranges
from vaxflux._interventions import Implementation, Intervention
from vaxflux._vaxflux_inference_data import VaxfluxInferenceData
from vaxflux._vaxflux_model import VaxfluxModel
from vaxflux._vaxflux_observations import VaxfluxObservations
