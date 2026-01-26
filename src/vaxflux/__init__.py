"""Model seasonal vaccination uptake curves."""

__all__ = (
    "Covariate",
    "Curve",
    "Implementation",
    "Intervention",
    "LogisticCurve",
    "PartiallyPooledGaussianCovariate",
    "PooledGaussianCovariate",
    "VaxfluxInferenceData",
    "VaxfluxModel",
    "covariates",
    "curves",
    "data",
    "dates",
    "interventions",
    "uptake",
)
__version__ = "0.2.0"


from vaxflux import (
    covariates,
    curves,
    data,
    dates,
    interventions,
    uptake,
)
from vaxflux._covariates import (
    Covariate,
    PartiallyPooledGaussianCovariate,
    PooledGaussianCovariate,
)
from vaxflux._curves import Curve, LogisticCurve
from vaxflux._interventions import Implementation, Intervention
from vaxflux._vaxflux_inference_data import VaxfluxInferenceData
from vaxflux._vaxflux_model import VaxfluxModel
