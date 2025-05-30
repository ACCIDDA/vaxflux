Getting Started
===============

This tutorial will guide you through the basic steps to get started with the `vaxflux` package, which is designed for modeling vaccine uptake prevalence using various curve families. This particular tutorial introduces the core concepts of the package, generates a synthetic dataset, how to fit a model to this data, and steps for working with a fitted model.

Selecting A Curve Family
------------------------

At the core of `vaxflux` is the concept of a curve family. A curve family is a set of curves that share a common structure and are used for modeling the uptake prevalence of a vaccine. This package provides several curve families via the :obj:`vaxflux.curves` module, including logistic which we'll use in this example. Advanced users can also create their own curve families to model different types of vaccine uptake patterns.

.. code-block:: python

    from vaxflux.curves import LogisticCurve
    logistic_curve = LogisticCurve()

Season And Date Ranges
----------------------

In `vaxflux`, a season is defined as a period of time during which vaccine uptake is measured. The package allows you to specify the start and end dates of a season, which is crucial for modeling vaccine uptake patterns over time. These are specified to the model using the :obj:`vaxflux.dates.SeasonRange` class. Within each season, you can define a date range that represents the period during which vaccine uptake is observed, which is done using the :obj:`vaxflux.dates.DateRange` class. For this example we will create a season that starts the 1st Monday of October and ends the 1st Sunday of the following February for the 2022/23, 2023/24, and 2024/25 seasons. Then we'll define date ranges for each of these seasons that span a week using the :func:`vaxflux.dates.daily_date_ranges` function.

.. code-block:: python

    from vaxflux.dates import DateRange, SeasonRange, daily_date_ranges
    seasons = [
        SeasonRange(season="2022/23", start="2022-10-03", end="2023-02-05"),
        SeasonRange(season="2023/24", start="2023-10-02", end="2024-02-04"),
        SeasonRange(season="2024/25", start="2024-10-07", end="2025-02-02"),
    ]
    dates = daily_date_ranges(seasons, range_days=7)

Defining Covariates
-------------------

In `vaxflux`, covariates are additional variables that can influence vaccine uptake patterns. These can include demographic information, geographic data, or any other relevant factors. Covariates and their categories are defined using the :obj:`vaxflux.covariates.CovariateCategories` class. In this example, we will create a single "age" covariate with three categories, "youth", "adult", and "elderly", which loosely correspond to 0-17 yrs, 18-65 yrs, and 65+ yrs, respectively. This covariate will be used to model how vaccine uptake varies across different age groups.

.. code-block:: python

    from vaxflux.covariates import CovariateCategories
    age_covariate = CovariateCategories(
        covariate="age",
        categories=["youth", "adult", "elderly"],
    )

Create A Sample Dataset
-----------------------

As a first step in using the package you can create a sample dataset with :func:`vaxflux.data.sample_dataset`. This will create a dataset with the same data generating process that the model assumes in the format needed for `vaxflux`.
