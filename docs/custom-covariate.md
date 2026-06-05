# Custom Covariate

This guide walks through implementing a custom covariate for use with `vaxflux`.
While `vaxflux` ships with
[`vaxflux.PooledGaussianCovariate`](api/vaxflux.md#vaxflux.PooledGaussianCovariate),
[`vaxflux.PartiallyPooledGaussianCovariate`](api/vaxflux.md#vaxflux.PartiallyPooledGaussianCovariate),
and
[`vaxflux.SeasonVaryingPartiallyPooledGaussianCovariate`](api/vaxflux.md#vaxflux.SeasonVaryingPartiallyPooledGaussianCovariate),
any prior structure can be expressed by implementing the
[`vaxflux.Covariate`](api/vaxflux.md#vaxflux.Covariate).

## Implementing the `Covariate` Class

All custom covariates inherit from `vaxflux.Covariate`:

```python
from vaxflux import Covariate
```

`Covariate` is a Pydantic `BaseModel`, so all configuration attributes are
declared as class-level fields and are validated automatically.

The only method you **must** implement is `sample`. It is called during model
construction and must register the appropriate numpyro sites.

```python
from vaxflux import Covariate, NumericalArrayLike


class MyCovariate(Covariate):
    def sample(
        self,
        *,
        season_names: list[str],
        covariate_categories: list[str] | None,
    ) -> NumericalArrayLike:
        ...
```

### The `sample` Method Contract

`sample` must register a `numpyro.deterministic` site named
`f"covariate_values_{self.prefix}"` and return its value. The required shape
depends on whether the covariate is **seasonal** or **categorical**:

| Kind        | `covariate` field    | `covariate_categories` argument | Required output shape           |
| ----------- | -------------------- | ------------------------------- | ------------------------------- |
| Seasonal    | `None`               | `None`                          | `(num_seasons,)`                |
| Categorical | `"age"` (any string) | `["<18", "18-64", "65+"]`       | `(num_seasons, num_categories)` |

For categorical covariates it's recommended that you constrain the values for
identifiability. With the covariates included with `vaxflux` this is done by
setting the effect for the index 0 category to zero.

The `prefix` property derives a unique site-name prefix from `parameter` and
`covariate`:

- Seasonal: `f"{parameter}_season"` - e.g. `"s_season"`
- Categorical: `f"{parameter}_{covariate}"` - e.g. `"s_age"`

### Registering Extra numpyro Sites

If `sample` registers additional named sites beyond `covariate_values_*` (such
as hyperparameter draws), override `extra_dims` to tell ArviZ how to label those
sites with coordinates:

```python
class MyCovariate(Covariate):
    def extra_dims(
        self,
        season_coord: str,
        category_short_coord: str | None,
    ) -> dict[str, list[str]]:
        return {
            f"{self.prefix}_mu": [category_short_coord or season_coord],
        }
```

Return an empty dict, the default, when there are no additional sites to label.

## Examples

### Student-_t_ Seasonal Covariate

As a concrete example, consider replacing the Gaussian seasonal prior with a
Student-$t$ distribution. This places heavier tails on the seasonal effects,
making the prior more robust to outlier seasons:

$$
x_i \sim \mathrm{StudentT}(\nu, \mu, \sigma)
$$

where $\nu > 0$ controls the tail weight ($\nu \to \infty$ recovers the
Gaussian), $\mu$ is the location, and $\sigma > 0$ is the scale.

```python
import numpyro
import numpyro.distributions as dist
from pydantic import Field
from typing import cast

from vaxflux import Covariate, NumericalArrayLike


class StudentTSeasonalCovariate(Covariate):
    r"""
    Seasonal covariate with a Student-t prior.

    $$
    x_i \sim \mathrm{StudentT}(\nu, \mu, \sigma)
    $$

    Heavier tails than the Gaussian make this prior more robust when one or
    more seasons are outliers.

    Attributes:
        nu: Degrees of freedom (> 0). Lower values give heavier tails.
        mu: Location parameter.
        sigma: Scale parameter (> 0).
    """

    nu: float = Field(gt=0.0)
    mu: float
    sigma: float = Field(gt=0.0)

    def sample(
        self,
        *,
        season_names: list[str],
        covariate_categories: list[str] | None,
    ) -> NumericalArrayLike:
        with numpyro.plate(f"covariate_{self.prefix}", len(season_names)):
            sampled_values = cast(
                "NumericalArrayLike",
                numpyro.sample(
                    self.prefix,
                    dist.StudentT(self.nu, self.mu, self.sigma),
                ),
            )
        return cast(
            "NumericalArrayLike",
            numpyro.deterministic(
                f"covariate_values_{self.prefix}", sampled_values
            ),
        )
```

Because this covariate only makes sense for seasonal use, no `covariate`
argument, `covariate_categories` will always be `None` at runtime and there is
no categorical branch needed.

Once defined, register it with the model exactly as you would the built-in
covariates:

```python
from vaxflux import VaxfluxModel

model = VaxfluxModel(curve=...)
model.add_covariates(
    StudentTSeasonalCovariate(parameter="s", nu=4.0, mu=40.0, sigma=10.0)
)
```

### Hierarchical Categorical Covariate with a Shared Scale

As a second example, consider a categorical covariate whose effects share a
single scale hyperparameter drawn from a prior rather than being fixed:

$$
\begin{aligned}
\sigma &\sim \mathrm{HalfNormal}(\tau) \\
x_k &\sim \mathrm{Normal}(0, \sigma), \quad k = 2, \dots, K
\end{aligned}
$$

The shared $\sigma$ pools information across categories while still letting each
category have its own effect.

```python
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from pydantic import Field
from typing import cast

from vaxflux import Covariate, NumericalArrayLike


class SharedScaleCategoricalCovariate(Covariate):
    r"""
    Categorical covariate with a shared, learned scale.

    $$
    \begin{aligned}
    \sigma &\sim \mathrm{HalfNormal}(\tau) \\
    x_k &\sim \mathrm{Normal}(0, \sigma)
    \end{aligned}
    $$

    All non-baseline categories share a single scale drawn from a
    HalfNormal prior, pooling information about effect magnitude.

    Attributes:
        tau: Scale of the HalfNormal prior on the shared standard deviation.
        covariate: Name of the covariate variable (required).
    """

    covariate: str
    tau: float = Field(gt=0.0)

    def extra_dims(
        self,
        season_coord: str,
        category_short_coord: str | None,
    ) -> dict[str, list[str]]:
        short = category_short_coord or season_coord
        return {
            self.prefix: [season_coord, short],
        }

    def sample(
        self,
        *,
        season_names: list[str],
        covariate_categories: list[str] | None,
    ) -> NumericalArrayLike:
        categories = cast("list[str]", covariate_categories)
        num_seasons = len(season_names)
        num_categories = len(categories)

        sigma = numpyro.sample(
            f"{self.prefix}_sigma", dist.HalfNormal(self.tau)
        )
        with numpyro.plate(f"covariate_{self.prefix}", num_categories - 1):
            sampled_values = cast(
                "NumericalArrayLike",
                numpyro.sample(self.prefix, dist.Normal(0.0, sigma)),
            )
        padded = jnp.pad(sampled_values, (1, 0), mode="constant")
        inflated = jnp.broadcast_to(
            padded[jnp.newaxis, :],
            (num_seasons, num_categories),
        )
        return cast(
            "NumericalArrayLike",
            numpyro.deterministic(
                f"covariate_values_{self.prefix}", inflated
            ),
        )
```

Register it alongside `CovariateCategories` so the model knows the category
labels:

```python
from vaxflux import CovariateCategories, VaxfluxModel

model = VaxfluxModel(curve=...)
model.add_covariate_categories(
    CovariateCategories(covariate="age", categories=["<18", "18-64", "65+"])
)
model.add_covariates(
    SharedScaleCategoricalCovariate(parameter="s", covariate="age", tau=5.0)
)
```
