Model Details
=============

This guide provides detailed information about the underlying model used in `vaxflux` for modeling vaccination uptake.

Curve Families
--------------

`vaxflux` starts with selecting a curve family. The curve family will be used to define the general shape of the cumulative vaccination uptake curve across all seasons and covariate combinations, but with different parameters. A curve, :math:`f(t\vert\theta_1,\dots,\theta_n)`, should have a few key properties:

- As :math:`t \rightarrow -\infty` then :math:`f(t \vert \theta_1, \dots, \theta_n) \rightarrow 0`.
- As :math:`t \rightarrow \infty` then :math:`f(t \vert \theta_1, \dots, \theta_n) \rightarrow L \leq 1`.
- The curve should be differentiable with respect to :math:`t` and :math:`\theta_i`.
- The curve should be monotonic with respect to :math:`t`.

`vaxflux` provides two curves, :obj:`vaxflux.curves.LogisticCurve` and :obj:`vaxflux.curves.TanhCurve` that satisfy these properties. The logistic curve is defined as:

.. math::

    f(t\vert m,r,s) = \mathrm{invlogit}\left(m\right)\mathrm{logit}\left(e^r\left(t-s\right)\right)

where :math:`\mathrm{invlogit}(m)` is the max uptake of the curve, :math:`r` is the rate of uptake, and :math:`s` is the inflection point of the curve. The logistic curve is a common choice for modeling vaccination uptake as it captures the typical S-shaped curve seen in many vaccination campaigns.

The tanh curve is defined as:

.. math::

    f(t\vert m,r,s) = \frac{1}{2}\mathrm{invlogit}\left(m\right)\left(\tanh\left(e^r\left(t-s\right)\right)+1\right)

where :math:`\mathrm{invlogit}(m)`, :math:`r`, and :math:`s` are defined the same as the logistic curve. The tanh curve is similar to the logistic curve but has a different steep-ness in it's S-shape, which can be useful for modeling different types of vaccination uptake patterns.

Curve Parameters
----------------

While the same underlying curve family is used for all seasons and covariate combinations, the parameters of the curve can vary. The parameters of the curve are determined by summing the seasonal baseline and the covariate effects.

For example, suppose we have a logistic curve and want to model the inflection point, :math:`s`, as being affected by a seasonal baseline, an age covariate, and a population density (rural or urban flag) covariate. We'll define these as being drawn from a normal distribution, but `vaxflux` allows for any distribution to be used. The inflection point can be defined as:

.. math::

    s^{\mathrm{season}}_{i}&\sim\mathcal{N}\left(\mu^{\mathrm{season}},\sigma^{\mathrm{season}^2}\right)

    s^{\mathrm{age}}_{i,j}&\sim\mathcal{N}\bf\left({\mu}^{\mathrm{age}},\Sigma^{\mathrm{age}^2}\right)

    s^{\mathrm{density}}_{i,k}&\sim\mathcal{N}\left({\mu}^{\mathrm{density}},\sigma^{\mathrm{density}^2}\right)

    s_{i,j,k} &= s^{\mathrm{season}}_{i} + s^{\mathrm{age}}_{i,j} + s^{\mathrm{density}}_{i,k}

Where :math:`i` is the season index, :math:`j` is the age category index, and :math:`k` is the population density category index. The :math:`s^{\mathrm{age}}` is drawn from a multivariate normal distribution and if population density had more than two categories it too would also be drawn from a multivariate normal distribution.

For covariates besides the seasonal baseline the first index is set to zero, so in our example :math:`s^{\mathrm{age}}_{i,1}=s^{\mathrm{density}}_{i,1}=0` (this is why for two categories the effect is not drawn from a multivariate distribution). The covariate indexes are determined by the order of `categories` provided to :obj:`vaxflux.covariates.CovariateCategories`. This means the interpretation of a covariate effect like :math:`s^{\mathrm{age}}_{i,j}` is that it is the difference in the inflection point between the :math:`j`-th age category and the first age category (which is the reference category).

A limitation of this approach to curve parameters is that it does not allow for interactions between covariates. For example, if we wanted to model the inflection point as being affected by the combination of age and population density, we would need to define a separate parameter for each combination of age and population density. This can lead to a large number of parameters and may not be practical (inability for fitting to converge, computational infeasibility, etc.) for most situations.

Observational Model
-------------------

Up until now this documentation has been describing the underlying model that defines the shape of the vaccination uptake curve. However, in practice we do not observe the true vaccination uptake curve, but rather a noisy version of it. The observational model is used to define how the observed data is generated from the underlying model.

The Model From The 'Getting Started' Example
--------------------------------------------

Recall from the :doc:`getting-started` guide that we defined a model with three age categories that affected the max uptake and a pooled rate of uptake and inflection point.
