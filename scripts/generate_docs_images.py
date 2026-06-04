"""Generate static images embedded in the documentation."""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl

mpl.use("Agg")

from vaxflux import Curve, LogisticCurve
from vaxflux._types import NumericalArrayLike

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "images"


class AlgebraicSigmoidCurve(Curve):
    """Algebraic sigmoid uptake curve for the custom curve documentation guide."""

    def prevalence(  # type: ignore[override]
        self,
        t: NumericalArrayLike,
        a: NumericalArrayLike,
        b: NumericalArrayLike,
        m: NumericalArrayLike,
    ) -> jax.Array:
        """Compute the algebraic sigmoid prevalence at time `t`."""
        u = jnp.exp(a) * (jnp.asarray(t) - b)
        return jnp.exp(m) / 2.0 * (u / jnp.sqrt(1.0 + u**2) + 1.0)

    def incidence(  # type: ignore[override]
        self,
        t: NumericalArrayLike,
        a: NumericalArrayLike,
        b: NumericalArrayLike,
        m: NumericalArrayLike,
    ) -> jax.Array:
        """Compute the algebraic sigmoid incidence at time `t`."""
        u = jnp.exp(a) * (jnp.asarray(t) - b)
        return jnp.exp(m) / 2.0 * jnp.exp(a) / (1.0 + u**2) ** 1.5


def getting_started_curve_plot(output_dir: Path) -> None:
    """Generate and save the LogisticCurve plot for the getting started guide."""
    curve = LogisticCurve()
    t = jnp.linspace(-10.0, 110.0, num=200)
    fig = curve.plot(
        t,
        parameter_sets=[
            {"m": 0.5, "r": -3.2, "s": 40.0},
            {"m": 1.2, "r": -3.2, "s": 40.0},
            {"m": 0.5, "r": -2.5, "s": 40.0},
            {"m": 0.5, "r": -3.2, "s": 20.0},
        ],
        labels=["base", "high m", "high r", "low s"],
        title="LogisticCurve",
    )
    fig.savefig(
        output_dir / "getting-started-curve-plot.png", dpi=150, bbox_inches="tight"
    )


def custom_curve_plot(output_dir: Path) -> None:
    """Generate and save the AlgebraicSigmoidCurve plot for the custom curve guide."""
    curve = AlgebraicSigmoidCurve()
    t = jnp.linspace(-10.0, 110.0, num=200)
    fig = curve.plot(
        t,
        parameter_sets=[
            {"a": -3.0, "b": 40.0, "m": 0.0},
            {"a": -2.0, "b": 40.0, "m": 0.0},
            {"a": -1.0, "b": 40.0, "m": 0.0},
        ],
        labels=["slow", "moderate", "steep"],
        title="AlgebraicSigmoidCurve",
    )
    fig.savefig(output_dir / "custom-curve-plot.png", dpi=150, bbox_inches="tight")


def main() -> None:
    """Generate all static images for the documentation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    getting_started_curve_plot(OUTPUT_DIR)
    custom_curve_plot(OUTPUT_DIR)


if __name__ == "__main__":
    main()
