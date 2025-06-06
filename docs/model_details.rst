Model Details
=============

This guide provides detailed information about the underlying model used in `vaxflux` for modeling vaccination uptake.

Curve Families
--------------

`vaxflux` starts with selecting a curve family. The curve family will be used to define the general shape of the vaccination uptake curve across all seasons and covariate combinations, but with different parameters. A curve, :math:`f(t\vert\theta_1,\dots,\theta_n)` should have a few key properties:

- As :math:`t \rightarrow -\infty` then :math:`f(t \vert \theta_1, \dots, \theta_n) \rightarrow 0`.
- As :math:`t \rightarrow \infty` then :math:`f(t \vert \theta_1, \dots, \theta_n) \rightarrow L \leq 1`.
- The curve should be differentiable with respect to :math:`t` and :math:`\theta_i`.
- The curve should be monotonic with respect to :math:`t`.

`vaxflux` provides two curves, :obj:`vaxflux.curves.LogisticCurve` and :obj:`vaxflux.curves.TanhCurve` that satisfy these properties. The logistic curve is defined as:

.. math::

    f(t \vert m, r, s) = \mathrm{invlogit}(m) r e^{-r(t-s)} \left( 1 + e^{-r(t-s)} \right)^{-2}

where :math:`\mathrm{invlogit}(m)` is the max uptake of the curve, :math:`r` is the rate of uptake, and :math:`s` is the inflection point of the curve. The logistic curve is a common choice for modeling vaccination uptake as it captures the typical S-shaped curve seen in many vaccination campaigns.

The tanh curve is defined as:

.. math::

    f(t \vert m, r, s) = \mathrm{invlogit}(m) \tanh\left( e^r (t - s) \right)

where :math:`\mathrm{invlogit}(m)`, :math:`r`, and :math:`s` are defined the same as the logistic curve. The tanh curve is similar to the logistic curve but has a different steep-ness in it's S-shape, which can be useful for modeling different types of vaccination uptake patterns.
