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


from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logit


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

    @abstractmethod
    def incidence(
        self,
        t: npt.NDArray[np.float64] | pt.variable.TensorVariable[Any, Any],
        **kwargs: npt.NDArray[np.float64]
        | pt.variable.TensorVariable[Any, Any]
        | float,
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
        **kwargs: npt.NDArray[np.float64]
        | pt.variable.TensorVariable[Any, Any]
        | float,
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
        **kwargs: npt.NDArray[np.float64]
        | pt.variable.TensorVariable[Any, Any]
        | float,
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
            "pt.variable.TensorVariable[Any, Any]",
            self.prevalence(t1, **kwargs) - self.prevalence(t0, **kwargs),
        )

    def propose_p0(
        self, t: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> tuple[float, ...]:
        """
        Propose an initial guess for the model parameters given a single timeseries.

        Args:
            t: The time steps to evaluate the incidence curve at.
            y: The prevalence values at the time steps provided.

        Returns:
            A proposed initial value for the prevalence at time zero with each entry
            corresponding to a parameter in the `parameters` attribute.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.

        """
        raise NotImplementedError

    def get_p0_proposal(
        self, t: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> tuple[float, ...]:
        """
        Get a proposed initial guess for the model parameters given a single timeseries.

        Args:
            t: The time steps to evaluate the incidence curve at.
            y: The prevalence values at the time steps provided.

        Returns:
            A proposed initial value for the prevalence at time zero.

        Raises:
            ValueError: If the length of `t` and `y` do not match.
            ValueError: If the length of `t` is less than the number of parameters
                plus one.
            ValueError: If the time steps or prevalence values are not monotonically
                increasing.
            ValueError: If the length of `p0` does not match the number of parameters
                in the `parameters` attribute.

        """
        if len(t) != len(y):
            msg = f"Expected {len(t)} time steps, got {len(y)} prevalence values."
            raise ValueError(msg)
        if len(t) < len(self.parameters) + 1:
            msg = (
                f"Expected at least {len(self.parameters) + 1} "
                f"time steps, got {len(t)}."
            )
            raise ValueError(msg)
        if np.any(np.diff(t) < 0) or np.any(np.diff(y) < 0):
            msg = (
                "The time steps and prevalence values must be monotonically increasing."
            )
            raise ValueError(msg)
        p0 = self.propose_p0(t, y)
        if len(p0) != len(self.parameters):
            msg = f"Expected {len(self.parameters)} parameters, got {len(p0)}."
            raise ValueError(msg)
        return p0


class LogisticCurve(Curve):
    r"""
    Logistic uptake curve.

    This class implements a logistic curve with parameters :math:`m`, :math:`r`, and
    :math:`s` which is given by:

    .. math::

        f(t\vert m,r,s)=\mathrm{invlogit}\left(m\right)\mathrm{logit}\left(e^r\left(t-s\right)\right)

    """  # noqa: E501

    #: The names of parameters used by this incidence curve model.
    parameters = ("m", "r", "s")

    def prevalence(
        self,
        t: npt.NDArray[np.float64] | pt.variable.TensorVariable[Any, Any],
        **kwargs: npt.NDArray[np.float64]
        | pt.variable.TensorVariable[Any, Any]
        | float,
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
            "pt.variable.TensorVariable[Any, Any]",
            pm.math.invlogit(kwargs["m"])
            * pm.math.invlogit(pm.math.exp(kwargs["r"]) * (t - kwargs["s"])),
        )

    def incidence(
        self,
        t: npt.NDArray[np.float64] | pt.variable.TensorVariable[Any, Any],
        **kwargs: npt.NDArray[np.float64]
        | pt.variable.TensorVariable[Any, Any]
        | float,
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
            "pt.variable.TensorVariable[Any, Any]",
            (
                pm.math.invlogit(kwargs["m"])
                * pm.math.exp(kwargs["r"])
                * pm.math.exp(u)
                * ((1 + pm.math.exp(u)) ** -2.0)
            ),
        )

    def propose_p0(
        self, t: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> tuple[float, ...]:
        r"""
        Propose an initial guess for the model parameters given a single timeseries.

        Args:
            t: The time steps to evaluate the incidence curve at.
            y: The prevalence values at the time steps provided.

        Returns:
            A proposed initial value for the prevalence at time zero with each entry
            corresponding to a parameter in the `parameters` attribute.

        Notes:
            This method proposes an initial guess for the parameters of the logistic
            curve by:

            - Taking the second largest value of the prevalence as the initial value
                for the parameter :math:`\\mathrm{invlogit}\\left(m\right)`.
            - Finding the time step closest to the midpoint of the prevalence values
                as the initial value for :math:`s`.
            - Taking the slope at the inflection point as the initial value for
                :math:`e^r`.

            This initial proposal is a rough heuristic and may not be optimal for all
            datasets. It is recommended to use this as a starting point and refine the
            parameters using a fitting procedure.

        """
        # The second largest value is used as the initial value for m.
        m = logit(np.partition(y, -2)[-2])
        # The inflection point is the time closest to the prevalence values midpoint.
        t_idx = np.argmin(np.abs(y - (0.5 * (np.max(y) - np.min(y))) + np.min(y)))
        s = t[t_idx]
        # Take the slope at the inflection point as the initial value for r.
        r = np.log(np.gradient(y, t)[t_idx])
        return (m, r, s)


class TanhCurve(Curve):
    r"""
    Tanh uptake curve.

    This class implements a :math:`\tanh` curve with parameters :math:`m`, :math:`r`,
    and :math:`s` which is given by:

    .. math::

        f(t\vert m,r,s)=\frac{1}{2}\mathrm{invlogit}\left(m\right)\left(\tanh\left(e^r\left(t-s\right)\right)+1\right)

    """  # noqa: E501

    #: The names of parameters used by this incidence curve model.
    parameters = ("m", "r", "s")

    def prevalence(
        self,
        t: npt.NDArray[np.float64] | pt.variable.TensorVariable[Any, Any],
        **kwargs: npt.NDArray[np.float64]
        | pt.variable.TensorVariable[Any, Any]
        | float,
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
            "pt.variable.TensorVariable[Any, Any]",
            0.5
            * pm.math.invlogit(kwargs["m"])
            * (pm.math.tanh(pm.math.exp(kwargs["r"]) * (t - kwargs["s"])) + 1.0),
        )

    def incidence(
        self,
        t: npt.NDArray[np.float64] | pt.variable.TensorVariable[Any, Any],
        **kwargs: npt.NDArray[np.float64]
        | pt.variable.TensorVariable[Any, Any]
        | float,
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
            "pt.variable.TensorVariable[Any, Any]",
            0.5
            * pm.math.invlogit(kwargs["m"])
            * pm.math.exp(kwargs["r"])
            * (pm.math.cosh(pm.math.exp(kwargs["r"]) * (t - kwargs["s"])) ** -2.0),
        )
