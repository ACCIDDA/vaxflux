"""Model seasonal vaccination uptake curves."""

__all__ = (
    "Curve",
    "LogisticCurve",
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
from vaxflux._curves import Curve, LogisticCurve
from vaxflux._vaxflux_model import VaxfluxModel
