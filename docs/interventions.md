# Interventions & Implementations

This guide explains how `vaxflux` models the effect of discrete interventions
such as a targeted outreach campaign, a policy change, etc.

## Concepts

`vaxflux` separates the what from the when and where using two objects:

- An **`Intervention`** defines a named effect and the prior distribution placed
  on that effect. It says "this type of event shifts a curve parameter by some
  unknown amount."
- An **`Implementation`** says when and where a specific intervention occurred:
  which season, which date range, and optionally which covariate categories.

One `Intervention` can have many `Implementation`s. For example, a single
`Intervention` named `"outreach"` might have one `Implementation` for the
2022/2023 season and another for the 2023/2024 season, each covering a different
date window.

## Mathematical Model

Each intervention shifts the value of a curve parameter on a daily basis. Let
$\theta_{t,i,\mathbf{c}}$ be the value of some curve parameter (e.g. the
inflection point $s$) on day $t$ in season $i$ for covariate combination
$\mathbf{c}$, before any interventions are applied. After applying all
implementations of all interventions that are active on day $t$, the adjusted
parameter is:

$$
\tilde{\theta}_{t,i,\mathbf{c}} = \theta_{t,i,\mathbf{c}} + \sum_{k} \delta_k \cdot \mathbf{1}\left[t \in [t^{\text{start}}_k, t^{\text{end}}_k]\right]
$$

where the sum is over every implementation $k$ whose season, date range, and
covariate filter all match $(t, i, \mathbf{c})$, and $\delta_k$ is the effect
sampled for that implementation from the intervention's prior. When multiple
implementations of the same intervention are active, `vaxflux` draws all of
their effects simultaneously using a
[`numpyro.plate`](https://num.pyro.ai/en/latest/primitives.html#plate), so
implementation effects are independent draws from the same prior.

The adjusted $\tilde{\theta}_{t,i,\mathbf{c}}$ is then passed to the curve in
place of $\theta_{t,i,\mathbf{c}}$ when computing daily incidence.

## Defining an Intervention

An [`Intervention`](api/vaxflux.md#vaxflux.Intervention) specifies:

- `name`: a lowercase alphanumeric identifier.
- `parameter`: which curve parameter the effect is added to (e.g. `"s"` for the
  inflection point of a `LogisticCurve`).
- `distribution`: the name of a
  [`numpyro.distributions`](https://num.pyro.ai/en/latest/distributions.html)
  class (e.g. `"Normal"`, `"HalfNormal"`, `"Exponential"`).
- `distribution_kwargs`: keyword arguments forwarded to that distribution's
  constructor.

```python
from vaxflux import Intervention

outreach = Intervention(
    name="outreach",
    parameter="s",
    distribution="Normal",
    distribution_kwargs={"loc": 0.0, "scale": 5.0},
)
```

## Defining an Implementation

An [`Implementation`](api/vaxflux.md#vaxflux.Implementation) specifies where and
when the intervention is deployed:

- `intervention`: the `name` of the `Intervention` this is an instance of.
- `season`: the season label (must match an added
  [`SeasonRange`](api/vaxflux.md#vaxflux.SeasonRange)).
- `start_date` / `end_date`: the inclusive date window the effect is active. Set
  either to `None` to extend to the season boundary.
- `covariate_categories`: an optional `{covariate: category}` dict restricting
  which covariate combinations the effect applies to. `None` means all
  combinations.

```python
from datetime import date
from vaxflux import Implementation

impl_2023 = Implementation(
    intervention="outreach",
    season="2023/2024",
    start_date=date(2023, 12, 1),
    end_date=date(2024, 1, 31),
    covariate_categories=None,
)
```

## Adding to the Model

Both `Intervention`s and `Implementation`s must be registered with the model
before sampling. Interventions must be added before implementations.

```python
from vaxflux import VaxfluxModel, LogisticCurve, SeasonRange

model = (
    VaxfluxModel(curve=LogisticCurve())
    .add_seasons(
        SeasonRange(season="2023/2024", start_date="2023-09-01", end_date="2024-05-31")
    )
    .add_interventions(outreach)
    .add_implementations(impl_2023)
)
```

## Choosing a Prior

The prior on the intervention effect is the primary way to encode your beliefs
about how an intervention influences the parameter - and to control what
hypotheses the model can express.

### Symmetric prior (effect could be positive or negative)

Use a zero-centred symmetric distribution, such as `Normal`, when you have no
reason to rule out a negative effect. This allows the data to push the posterior
in either direction and is appropriate when you want to _test_ whether the
intervention had any effect at all:

```python
Intervention(
    name="outreach",
    parameter="s",
    distribution="Normal",
    distribution_kwargs={"loc": 0.0, "scale": 5.0},
)
```

With this prior, a posterior credible interval that contains zero is consistent
with the intervention having had no effect.

### Positive-only prior (constrained to help)

Use a strictly positive distribution, such as `HalfNormal` or `Exponential`,
when you are certain the intervention can only shift the parameter in one
direction and want to encode that knowledge:

```python
# Shift the inflection point later (delay uptake) only
Intervention(
    name="supply_delay",
    parameter="s",
    distribution="HalfNormal",
    distribution_kwargs={"scale": 5.0},
)
```

A positive-only prior rules out a zero or negative effect by construction - the
posterior mass can never reach zero. If your goal is to _test_ whether an
intervention had any effect, or whether the effect could be harmful, you must
use a prior that assigns probability to zero or negative values. A `HalfNormal`
or `Exponential` prior is a strong modelling assumption, not a neutral one.

## Restricting to Covariate Subgroups

Implementations can target a specific covariate category combination using
`covariate_categories`. This is useful when an intervention was only deployed
for a particular demographic group:

```python
Implementation(
    intervention="outreach",
    season="2023/2024",
    start_date=None,
    end_date=None,
    covariate_categories={"age": "65+"},
)
```

Implementations with `covariate_categories=None` apply to every covariate
category combination, while a targeted implementation only modifies the
parameter for the rows whose covariate values match the filter exactly.

## Multiple Implementations of the Same Intervention

When an intervention has more than one implementation (e.g. it was deployed in
multiple seasons or to multiple subgroups), each implementation draws its own
effect from the shared prior independently:

```python
model.add_implementations(
    Implementation(
        intervention="outreach",
        season="2022/2023",
        start_date=date(2022, 12, 1),
        end_date=date(2023, 1, 31),
        covariate_categories=None,
    ),
    Implementation(
        intervention="outreach",
        season="2023/2024",
        start_date=date(2023, 12, 1),
        end_date=date(2024, 1, 31),
        covariate_categories=None,
    ),
)
```

The two draws ($\delta_0$ and $\delta_1$) are exchangeable. Their magnitudes can
differ, but both are drawn from the same `Normal(0, 5)` prior. This is
equivalent to partial pooling across implementations.
