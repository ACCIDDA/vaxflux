from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
import pymc as pm
import pytensor as pt


class IncidenceCurve(ABC):
    """
    Abstract class for implementations of incidence curves.
    """

    @abstractmethod
    def get_parameters(self) -> tuple[str]:
        """
        Return the set of parameters used by this incidence curve model.

        Returns:
            The set of parameter names as strings.
        """

    @abstractmethod
    def evaluate(
        t: npt.NDArray[np.number] | pt.tensor.variable.TensorVariable,
        **kwargs: npt.NDArray[np.number] | pt.tensor.variable.TensorVariable,
    ) -> npt.NDArray[np.number] | pt.tensor.variable.TensorVariable:
        """
        Evaluate the incidence curve at given set of time steps.

        Args:
            t: The time steps to evaluate the incidence curve at.
            kwargs: Further keyword arguments, must be the parameters for this model as
                returned by the `get_parameters` method.

        Returns:
            The incidence curve at the time steps provided.
        """


class LogisticIncidenceCurve(IncidenceCurve):
    """
    Logistic incidence curve.

    This class implements a logistic incidence curve with parameters $m$, $r$, and $s$
    which is given by:

    $$ f(t \vert m, r, s) = \mathrm{expit}(m) r e^{-r(t-s)} \left( 1 + e^{-r(t-s)} \right)^{-2}. $$
    """

    def get_parameters(self) -> tuple[str]:
        return ("m", "r", "s")

    def evaluate(
        self,
        t: npt.NDArray[np.number] | pt.tensor.variable.TensorVariable,
        m: npt.NDArray[np.number] | pt.tensor.variable.TensorVariable,
        r: npt.NDArray[np.number] | pt.tensor.variable.TensorVariable,
        s: npt.NDArray[np.number] | pt.tensor.variable.TensorVariable,
    ) -> pt.tensor.variable.TensorVariable:
        tmp = pm.math.exp(-r * (t - s))
        return (pm.math.invlogit(m) * r * tmp) * ((1.0 + tmp) ** (-2.0))
