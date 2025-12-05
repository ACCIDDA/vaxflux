__all__: list[str] = []

import inspect
from abc import ABC, abstractmethod
from typing import cast

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

from vaxflux._types import NumericalArrayLike


class Curve(ABC):
    """Abstract class for implementations of uptake curves."""

    def __init__(self) -> None:
        in_axes = (0,) + (None,) * (
            len(inspect.signature(self.prevalence).parameters) - 1
        )
        self._grad_prevalence = jax.vmap(jax.grad(self.prevalence), in_axes=in_axes)

    @abstractmethod
    def prevalence(
        self, t: NumericalArrayLike, **kwargs: NumericalArrayLike
    ) -> jax.Array:
        """
        Compute the prevalence at time `t`.

        Args:
            t: The time steps to evaluate the prevalence curve at.
            **kwargs: Additional parameters required by the prevalence model.

        Returns:
            The prevalence curve evaluated at time `t` bounded between 0 and 1.
        """
        raise NotImplementedError

    def incidence(
        self, t: NumericalArrayLike, **kwargs: NumericalArrayLike
    ) -> jax.Array:
        """
        Compute the incidence at time `t`.

        Args:
            t: The time steps to evaluate the incidence curve at.
            **kwargs: Additional parameters required by the incidence model.

        Returns:
            The incidence curve evaluated at time `t`.
        """
        return cast("jax.Array", self._grad_prevalence(*((t, *tuple(kwargs.values())))))


class LogisticCurve(Curve):
    r"""
    Logistic uptake curve implementation.

    The logistic curve is defined by the following prevalence function:

    This class implements a logistic curve with parameters $m$, $r$, and $s$ which is
    given by:

    $$ f(t\vert m,r,s)=\mathrm{invlogit}\left(m\right)\mathrm{logit}\left(e^r\left(t-s\right)\right) $$

    Examples:
        >>> import jax.numpy as jnp
        >>> from vaxflux import LogisticCurve
        >>> t = jnp.array([0.0, 1.0, 2.0, 3.0])
        >>> curve = LogisticCurve()
        >>> prevalence = curve.prevalence(t, m=0.0, r=1.0, s=1.0)
        >>> prevalence
        Array([0.03095159, 0.25      , 0.4690484 , 0.4978322 ], dtype=float32)
        >>> incidence = curve.incidence(t, m=0.0, r=1.0, s=1.0)
        >>> incidence
        Array([0.0789269 , 0.33978522, 0.07892691, 0.00586712], dtype=float32)

    """  # noqa: E501

    def prevalence(  # type: ignore[override]
        self,
        t: NumericalArrayLike,
        m: NumericalArrayLike,
        r: NumericalArrayLike,
        s: NumericalArrayLike,
    ) -> jax.Array:
        """
        Compute the logistic prevalence at time `t`.

        Args:
            t: The time steps to evaluate the prevalence curve at.
            m: The curve's maximum value.
            r: The steepness of the curve.
            s: The x-value of the sigmoid's midpoint.

        Returns:
            The logistic prevalence curve evaluated at time `t`.
        """
        return jsp.expit(m) * jsp.expit(jnp.exp(r) * (t - s))
