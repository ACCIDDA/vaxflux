"""
Generic incidence curve for vaccine uptake.

This module provides a generic interface for incidence curves used in the context of
vaccine uptake. The module provides an abstract base class `IncidenceCurve` which
defines the interface for incidence curve models. The module also provides a number of
concrete implementations of incidence curve models, including `LogisticIncidenceCurve`
which is most commonly used in the context of vaccine uptake.

"""

__all__ = (
    "GeneralizedLogisticIncidenceCurve",
    "IncidenceCurve",
    "LogisticIncidenceCurve",
    "TanhIncidenceCurve",
)


from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import pymc as pm
import pytensor as pt


class IncidenceCurve(ABC):
    """Abstract class for implementations of incidence curves."""

    @property
    @abstractmethod
    def parameters(self) -> tuple[str, ...]:
        """
        Return the set of parameters used by this incidence curve model.

        Returns:
            The set of parameter names as strings.

        Notes:
            1) The order of the parameters should be the same as the order of the
               parameters in the `evaluate` method.
            2) By convention the parameters should be named in lower case and preferably
               a single letter.
            3) When implementing a new incidence curve model, the `parameters` property
               can be specified as a tuple of strings.

        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        t: npt.NDArray[np.number] | pt.tensor.variable.TensorVariable,
        **kwargs: npt.NDArray[np.number] | pt.tensor.variable.TensorVariable,
    ) -> pt.tensor.variable.TensorVariable:
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

    def prevalence(
        self,
        t: npt.NDArray[np.number] | pt.tensor.variable.TensorVariable,
        **kwargs: npt.NDArray[np.number] | pt.tensor.variable.TensorVariable,
    ) -> pt.tensor.variable.TensorVariable:
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


class LogisticIncidenceCurve(IncidenceCurve):
    r"""
    Logistic incidence curve.

    This class implements a logistic incidence curve with parameters $m$, $r$, and $s$
    which is given by:

    $$ f(t \vert m, r, s) = \mathrm{expit}(m) r e^{-r(t-s)} \left( 1 + e^{-r(t-s)} \right)^{-2}. $$

    Attributes:
        parameters: The names of parameters used by this incidence curve model.

    """

    parameters = ("m", "r", "s")

    def evaluate(
        self,
        t: npt.NDArray[np.number] | pt.tensor.variable.TensorVariable,
        **kwargs: npt.NDArray[np.number] | pt.tensor.variable.TensorVariable,
    ) -> pt.tensor.variable.TensorVariable:
        """
        Evaluate the incidence curve at given set of time steps.

        Args:
            t: The time steps to evaluate the incidence curve at.
            kwargs: Further keyword arguments, must be the parameters for this model as
                described by the `parameters` attribute.

        Returns:
            The incidence curve at the time steps provided.

        """
        tmp = pm.math.exp(-pm.math.exp(kwargs["r"]) * (t - kwargs["s"]))
        return (
            pm.math.invlogit(kwargs["m"])
            * pm.math.exp(kwargs["r"])
            * tmp
            * ((1.0 + tmp) ** -2.0)
        )

    def prevalence(
        self,
        t: npt.NDArray[np.number] | pt.tensor.variable.TensorVariable,
        **kwargs: npt.NDArray[np.number] | pt.tensor.variable.TensorVariable,
    ) -> pt.tensor.variable.TensorVariable:
        """
        Evaluate the prevalence curve at given set of time steps.

        Args:
            t: The time steps to evaluate the prevalence curve at.
            kwargs: Further keyword arguments, must be the parameters for this model as
                described by the `parameters` attribute.

        Returns:
            The prevalence curve at the time steps provided.

        """
        return pm.math.invlogit(kwargs["m"]) * pm.math.invlogit(
            pm.math.exp(kwargs["r"]) * (t - kwargs["s"])
        )


class TanhIncidenceCurve(IncidenceCurve):
    r"""
    Tanh incidence curve.

    This class implements a tanh incidence curve with parameters $m$, $r$, and $s$ which
    is given by:

    $$ f(t \vert m, r, s) = \frac{r \mathrm{expit}(m)}{4}\mathrm{sech}^2\left( \frac{r}{2}(t - s) \right) $$

    Attributes:
        parameters: The names of parameters used by this incidence curve model.

    """

    parameters = ("m", "r", "s")

    def evaluate(
        self,
        t: npt.NDArray[np.number] | pt.tensor.variable.TensorVariable,
        **kwargs: npt.NDArray[np.number] | pt.tensor.variable.TensorVariable,
    ):
        """
        Evaluate the incidence curve at given set of time steps.

        Args:
            t: The time steps to evaluate the incidence curve at.
            kwargs: Further keyword arguments, must be the parameters for this model as
                described by the `parameters` attribute.

        Returns:
            The incidence curve at the time steps provided.

        """
        return (
            0.25
            * pm.math.invlogit(kwargs["m"])
            * kwargs["r"]
            * (pm.math.cosh(0.5 * kwargs["r"] * (t - kwargs["s"])) ** -2.0)
        )


class GeneralizedLogisticIncidenceCurve(IncidenceCurve):
    r"""
    Generalized logistic incidence curve.

    This class implements a generalized logistic incidence curve with parameters $m$,
    $r$, $s$, $q$, and $\lambda$ which is given by:

    $$ f(t \vert m, r, s, q, \lambda) = \frac{\mathrm{expit}(m)}{\left(1+e^{q - e^r(t - s)}\right)^{-\lambda}} $$.

    Attributes:
        parameters: The names of parameters used by this incidence curve model.

    """

    parameters = ("m", "r", "s", "q", "lam")

    def evaluate(
        self,
        t: npt.NDArray[np.number] | pt.tensor.variable.TensorVariable,
        **kwargs: npt.NDArray[np.number] | pt.tensor.variable.TensorVariable,
    ) -> pt.tensor.variable.TensorVariable:
        """
        Evaluate the incidence curve at given set of time steps.

        Args:
            t: The time steps to evaluate the incidence curve at.
            kwargs: Further keyword arguments, must be the parameters for this model as
                described by the `parameters` attribute.

        Raises:
            NotImplementedError: This method is not implemented for the Generalized
                Logistic Incidence Curve.

        """
        raise NotImplementedError

    def prevalence(
        self,
        t: npt.NDArray[np.number] | pt.tensor.variable.TensorVariable,
        **kwargs: npt.NDArray[np.number] | pt.tensor.variable.TensorVariable,
    ) -> pt.tensor.variable.TensorVariable:
        """
        Evaluate the prevalence curve at given set of time steps.

        Args:
            t: The time steps to evaluate the prevalence curve at.
            kwargs: Further keyword arguments, must be the parameters for this model as
                described by the `parameters` attribute.

        Returns:
            The prevalence curve at the time steps provided.

        """
        return pm.math.invlogit(kwargs["m"]) * (
            1.0
            + (
                pm.math.exp(kwargs["q"])
                * pm.math.exp(-pm.math.exp(kwargs["r"]) * (t - kwargs["s"]))
            )
        ) ** (-pm.math.exp(kwargs["lam"]))
