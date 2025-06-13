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

Observational Model
-------------------

Up until now this documentation has been describing the underlying model that defines the shape of the vaccination uptake curve. However, in practice we do not observe the true vaccination uptake curve, but rather a noisy version of it. The observational model is used to define how the observed data is generated from the underlying model.

The Model From The 'Getting Started' Example
--------------------------------------------

Recall from the :doc:`getting-started` guide that we defined a model with three age categories that affected the max uptake and a pooled rate of uptake and inflection point.
