# Custom Curve

This guide walks through implementing a custom uptake curve for use with
`vaxflux`. While `vaxflux` ships with
[`vaxflux.LogisticCurve`](api/vaxflux.md#vaxflux.LogisticCurve), any curve that
captures the right shape can be subclassed from
[`vaxflux.Curve`](api/vaxflux.md#vaxflux.Curve).

## Implementing the `Curve` Class

All custom curves inherit from `vaxflux.Curve`:

```python
from vaxflux import Curve
```

The only method you **must** implement is `prevalence`. Everything else
(`incidence`, `prevalence_difference`, `plot`) is derived automatically.

```python
import jax
import jax.numpy as jnp
from vaxflux import Curve
from vaxflux._types import NumericalArrayLike


class MyCurve(Curve):
    def prevalence(
        self, t: NumericalArrayLike, a: NumericalArrayLike, b: NumericalArrayLike
    ) -> jax.Array:
        ...
```

`Curve.__init__` inspects the signature of `prevalence` to discover parameter
names, so the positional arguments after `t` become `curve.parameters`
automatically.

### Overriding the `incidence` Method

By default, `incidence` is computed by differentiating `prevalence` with
`jax.grad`. If your prevalence function has a known closed-form derivative, you
can override `incidence` directly for better performance:

```python
class MyCurve(Curve):
    def prevalence(self, t, a, b):
        ...

    def incidence(self, t, a, b):
        # Closed-form derivative — faster than auto differentiation
        ...
```

The override must accept the same parameters as `prevalence`.

### Pretty Printing in Jupyter Notebooks

Jupyter notebooks call `_repr_latex_` to display objects as rendered math. Add
this method to your curve class so notebooks show the formula instead of the
default `repr`:

```python
class MyCurve(Curve):
    def _repr_latex_(self) -> str:
        return r"$$f(t \mid a, b) = \dots$$"
```

## Example: Algebraic Sigmoid Curve

As a concrete example, consider the algebraic sigmoid:

$$
\frac{x}{\sqrt{1 + x^2}}
$$

This function passes through the origin and asymptotes to $\pm 1$. Shifting it
up and scaling it to $[0, m]$ gives a well-behaved prevalence curve:

$$
f(t \mid a, b, m) = \frac{m}{2}\left(\frac{a(t-b)}{\sqrt{1+a^2(t-b)^2}} + 1\right)
$$

where:

- $a > 0$ controls the steepness of the curve,
- $b$ is the inflection point (the time at which uptake is at half its maximum),
- $m \in (0, 1]$ is the maximum uptake prevalence.

The derivative with respect to $t$ is:

$$
\frac{\partial f}{\partial t} = \frac{ma}{2} \cdot \frac{1}{\left(1 + a^2(t-b)^2\right)^{3/2}}
$$

which is straightforward to implement directly.

```python
import jax
import jax.numpy as jnp
from vaxflux import Curve
from vaxflux._types import NumericalArrayLike


class AlgebraicSigmoidCurve(Curve):
    r"""
    Algebraic sigmoid uptake curve.

    $$
    f(t \mid a, b, m) =
    \frac{m}{2}\left(\frac{a(t-b)}{\sqrt{1+a^2(t-b)^2}}+1\right)
    $$
    """

    def prevalence(  # type: ignore[override]
        self,
        t: NumericalArrayLike,
        a: NumericalArrayLike,
        b: NumericalArrayLike,
        m: NumericalArrayLike,
    ) -> jax.Array:
        u = jnp.exp(a) * (jnp.asarray(t) - b)
        return jnp.exp(m) / 2.0 * (u / jnp.sqrt(1.0 + u**2) + 1.0)

    def incidence(  # type: ignore[override]
        self,
        t: NumericalArrayLike,
        a: NumericalArrayLike,
        b: NumericalArrayLike,
        m: NumericalArrayLike,
    ) -> jax.Array:
        u = jnp.exp(a) * (jnp.asarray(t) - b)
        return jnp.exp(m) / 2.0 * jnp.exp(a) / (1.0 + u**2) ** 1.5

    def _repr_latex_(self) -> str:
        return (
            r"$$f(t \mid a, b, m) = "
            r"\frac{e^m}{2}\left("
            r"\frac{e^a(t-b)}{\sqrt{1+e^{2a}(t-b)^2}}+1"
            r"\right)$$"
        )
```

Note that $a$ and $m$ are passed in log-space (`jnp.exp(a)` and `jnp.exp(m)`) so
that the optimizer can treat them as unconstrained reals while the curve always
receives positive slope and max values. This mirrors how `LogisticCurve` handles
its own parameters.

You can verify the curve behaves as expected by plotting it:

```python
import jax.numpy as jnp

curve = AlgebraicSigmoidCurve()
t = jnp.linspace(-10.0, 110.0, num=200)

curve.plot(
    t,
    parameter_sets=[
        {"a": -3.0, "b": 40.0, "m": 0.0},   # slow
        {"a": -2.0, "b": 40.0, "m": 0.0},   # moderate
        {"a": -1.0, "b": 40.0, "m": 0.0},   # steep
    ],
    labels=["slow", "moderate", "steep"],
    title="AlgebraicSigmoidCurve",
)
```

![AlgebraicSigmoidCurve plot](images/custom-curve-plot.png)

Once the curve is defined, it is a drop-in replacement for `LogisticCurve`
anywhere a `Curve` is expected:

```python
from vaxflux import VaxfluxModel

model = VaxfluxModel(curve=AlgebraicSigmoidCurve())
```
