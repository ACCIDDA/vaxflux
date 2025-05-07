"""
Generic curve for vaccine uptake.

This module provides a generic interface for curves used in the context of vaccine
uptake. The module provides an abstract base class `Curve` which defines the interface
for curve models. The module also provides a number of concrete implementations of curve
models, including `LogisticCurve` which is most commonly used in the context of vaccine
uptake.
"""

__all__ = (
    "Curve",
    "LogisticCurve",
    "TanhCurve",
)


import warnings
from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pymc as pm
import pytensor.tensor as pt


class Curve(ABC):
    """Abstract class for implementations of uptake curves."""

    @property
    @abstractmethod
    def parameters(self) -> tuple[str, ...]:
        """
        Return the set of parameters used by this curve model.

        Returns:
            The set of parameter names as strings.

        Notes:
            1) The order of the parameters should be the same as the order of the
               parameters in the `evaluate` method.
            2) By convention the parameters should be named in lower case and preferably
               a single letter.
            3) When implementing a new curve model, the `parameters` property
               can be specified as a tuple of strings.

        """
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        """Deprecated in favor of the `incidence` method."""
        warnings.warn(
            "The `evaluate` method is deprecated, use `incidence` instead.",
            DeprecationWarning,
        )
        return self.incidence(*args, **kwargs)

    @abstractmethod
    def incidence(
        self,
        t: npt.NDArray[np.float64] | pt.variable.TensorVariable[Any, Any],
        **kwargs: npt.NDArray[np.float64] | pt.variable.TensorVariable[Any, Any],
    ) -> pt.variable.TensorVariable[Any, Any]:
        """
        Evaluate the incidence curve at given set of time steps.

        Args:
            t: The time steps to evaluate the incidence curve at.
            kwargs: Further keyword arguments, must be the parameters for this model as
                returned by the `get_parameters` method.

        Returns:
            The incidence curve at the time steps provided.

        """
        raise NotImplementedError

    @abstractmethod
    def prevalence(
        self,
        t: npt.NDArray[np.float64] | pt.variable.TensorVariable[Any, Any],
        **kwargs: npt.NDArray[np.float64] | pt.variable.TensorVariable[Any, Any],
    ) -> pt.variable.TensorVariable[Any, Any]:
        """
        Evaluate the prevalence curve at given set of time steps.

        Args:
            t: The time steps to evaluate the incidence curve at.
            kwargs: Further keyword arguments, must be the parameters for this model as
                returned by the `get_parameters` method.

        Returns:
            The incidence curve at the time steps provided.

        """
        raise NotImplementedError

    def prevalence_difference(
        self,
        t0: npt.NDArray[np.float64] | pt.variable.TensorVariable[Any, Any],
        t1: npt.NDArray[np.float64] | pt.variable.TensorVariable[Any, Any],
        **kwargs: npt.NDArray[np.float64] | pt.variable.TensorVariable[Any, Any],
    ) -> pt.variable.TensorVariable[Any, Any]:
        """
        Evaluate the difference in prevalence between two time steps.

        Args:
            t0: The first time step to evaluate the prevalence curve at.
            t1: The second time step to evaluate the prevalence curve at.
            kwargs: Further keyword arguments, must be the parameters for this model as
                returned by the `get_parameters` method.

        Returns:
            The difference in prevalence between the two time steps provided.

        """
        return cast(
            pt.variable.TensorVariable[Any, Any],
            self.prevalence(t1, **kwargs) - self.prevalence(t0, **kwargs),
        )


class LogisticCurve(Curve):
    r"""
    Logistic uptake curve.

    This class implements a logistic curve with parameters $m$, $r$, and $s$ which is
    given by:

    $$ f(t \vert m, r, s)
        = \mathrm{expit}(m) r e^{-r(t-s)} \left( 1 + e^{-r(t-s)} \right)^{-2}. $$

    Attributes:
        parameters: The names of parameters used by this incidence curve model.

    """

    parameters = ("m", "r", "s")

    def prevalence(
        self,
        t: npt.NDArray[np.float64] | pt.variable.TensorVariable[Any, Any],
        **kwargs: npt.NDArray[np.float64] | pt.variable.TensorVariable[Any, Any],
    ) -> pt.variable.TensorVariable[Any, Any]:
        """
        Evaluate the prevalence curve at given set of time steps.

        Args:
            t: The time steps to evaluate the prevalence curve at.
            kwargs: Further keyword arguments, must be the parameters for this model as
                described by the `parameters` attribute.

        Returns:
            The prevalence curve at the time steps provided.

        """
        return cast(
            pt.variable.TensorVariable[Any, Any],
            pm.math.invlogit(kwargs["m"])
            * pm.math.invlogit(pm.math.exp(kwargs["r"]) * (t - kwargs["s"])),
        )

    def incidence(
        self,
        t: npt.NDArray[np.float64] | pt.variable.TensorVariable[Any, Any],
        **kwargs: npt.NDArray[np.float64] | pt.variable.TensorVariable[Any, Any],
    ) -> pt.variable.TensorVariable[Any, Any]:
        """
        Evaluate the incidence curve at given set of time steps.

        Args:
            t: The time steps to evaluate the incidence curve at.
            kwargs: Further keyword arguments, must be the parameters for this model as
                described by the `parameters` attribute.

        Returns:
            The incidence curve at the time steps provided.

        """
        u = pm.math.exp(kwargs["r"]) * (t - kwargs["s"])
        return cast(
            pt.variable.TensorVariable[Any, Any],
            (
                pm.math.invlogit(kwargs["m"])
                * pm.math.exp(kwargs["r"])
                * pm.math.exp(u)
                * ((1 + pm.math.exp(u)) ** -2.0)
            ),
        )


class TanhCurve(Curve):
    r"""
    Tanh uptake curve.

    This class implements a tanh curve with parameters $m$, $r$, and $s$.

    Attributes:
        parameters: The names of parameters used by this curve model.

    """

    parameters = ("m", "r", "s")

    def prevalence(
        self,
        t: npt.NDArray[np.float64] | pt.variable.TensorVariable[Any, Any],
        **kwargs: npt.NDArray[np.float64] | pt.variable.TensorVariable[Any, Any],
    ) -> pt.variable.TensorVariable[Any, Any]:
        """
        Evaluate the prevalence curve at given set of time steps.

        Args:
            t: The time steps to evaluate the prevalence curve at.
            kwargs: Further keyword arguments, must be the parameters for this model as
                described by the `parameters` attribute.

        Returns:
            The prevalence curve at the time steps provided.

        """
        return cast(
            pt.variable.TensorVariable[Any, Any],
            pm.math.invlogit(kwargs["m"])
            * pm.math.tanh(pm.math.exp(kwargs["r"]) * (t - kwargs["s"])),
        )

    def incidence(
        self,
        t: npt.NDArray[np.float64] | pt.variable.TensorVariable[Any, Any],
        **kwargs: npt.NDArray[np.float64] | pt.variable.TensorVariable[Any, Any],
    ) -> pt.variable.TensorVariable[Any, Any]:
        """
        Evaluate the incidence curve at given set of time steps.

        Args:
            t: The time steps to evaluate the incidence curve at.
            kwargs: Further keyword arguments, must be the parameters for this model as
                described by the `parameters` attribute.

        Returns:
            The incidence curve at the time steps provided.

        """
        return cast(
            pt.variable.TensorVariable[Any, Any],
            (
                pm.math.invlogit(kwargs["m"])
                * pm.math.exp(kwargs["r"])
                * (pm.math.cosh(pm.math.exp(kwargs["r"]) * (t - kwargs["s"])) ** -2.0)
            ),
        )
