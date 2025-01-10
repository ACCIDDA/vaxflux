__all__ = ("build_model", "change_detection")


from collections.abc import Iterable
from typing import Any, Callable, Literal

from arviz import InferenceData
import numpy as np
import pandas as pd
import pymc as pm

from vaxflux.curves import IncidenceCurve
from vaxflux.data import coordinates_from_incidence


def build_model(
    data: pd.DataFrame,
    observation_type: Literal["incidence", "prevalence"],
    value_type: Literal["rate", "count"],
    incidence_curve: IncidenceCurve,
    observational_prior: pm.Distribution,
    epsilon_prior: tuple[pm.Distribution, dict[str, Any]],
    parameter_priors: dict[str, tuple[pm.Distribution, dict[str, Any]]],
    season_stratified_parameters: tuple[str] = (),
) -> pm.Model:
    """
    Build an model for vaccine incidence.

    Args:
        data: DataFrame with columns "time", "season", "region", "strata", and "value".
        observation_type: The type of observation, either "incidence" or "prevalence".
        value_type: The type of value, either "rate" or "count".
        incidence_curve: An `IncidenceCurve` to fit the uptake with.
        observational_prior: A PyMC distribution for the observational model.
        epsilon_prior: A tuple with the PyMC distribution for the epsilon parameter and
            its parameters.
        parameter_priors: A dictionary with the PyMC distribution for each parameter and
            its parameters.
        season_stratified_parameters: A tuple with the parameters that are stratified by
            season.

    Returns:
        A PyMC model object.
    """
    if observation_type != "incidence":
        raise NotImplementedError(
            "Only incidence observations is supported at the moment."
        )
    if value_type != "rate":
        raise NotImplementedError("Only rate values are supported at the moment.")

    # Determining the coordinates and parameters
    coords = coordinates_from_incidence(data)
    coords.update(
        {f"{p}Season": coords["season"] for p in season_stratified_parameters}
    )
    coords.update(
        {
            f"{p}Season": ["All Seasons"]
            for p in set(incidence_curve.parameters) - set(season_stratified_parameters)
        }
    )
    indexes = {
        column: data[column].apply(lambda x: coords[column].index(x)).values
        for column in ("season", "region", "strata")
    }
    indexes.update(
        {f"{p}Season": indexes["season"] for p in season_stratified_parameters}
    )
    indexes.update(
        {
            f"{p}Season": np.array([0])
            for p in set(incidence_curve.parameters) - set(season_stratified_parameters)
        }
    )

    # Build the model
    with pm.Model(coords=coords) as model:
        # **Data**
        time = pm.Data("time", data["time"].values)
        observed_incidence = pm.Data("observedIncidence", data["value"].values)

        # **Prior distributions**
        params = {}
        for p in incidence_curve.parameters:
            # Macro
            p_season = f"{p}Macro"
            dist, dist_params = parameter_priors[p]
            params[p_season] = dist(name=p_season, dims=f"{p}Season", **dist_params)
            # Region
            if (p_region := f"{p}Region") in parameter_priors:
                dist, dist_params = parameter_priors[p_region]
                params[p_region] = dist(name=p_region, dims="region", **dist_params)
            else:
                params[p_region] = pm.Data(
                    p_region, np.repeat(0.0, len(coords["region"])), dims="region"
                )
            # Strata
            if (p_strata := f"{p}Strata") in parameter_priors:
                dist, dist_params = parameter_priors[p_strata]
                params[p_strata] = dist(name=p_strata, dims="strata", **dist_params)
            else:
                params[p_strata] = pm.Data(
                    p_strata, np.repeat(0.0, len(coords["strata"])), dims="strata"
                )

        # **Model computations**
        evaluate_params = {
            p: pm.Deterministic(
                p,
                params[f"{p}Macro"][indexes[f"{p}Season"]]
                + params[f"{p}Region"][indexes["region"]]
                + params[f"{p}Strata"][indexes["strata"]],
                dims="observation",
            )
            for p in incidence_curve.parameters
        }
        y_model = pm.Deterministic(
            "yModel",
            incidence_curve.evaluate(time, **evaluate_params),
            dims="observation",
        )

        # **Observational model**
        epsilon = epsilon_prior[0](name="epsilon", **epsilon_prior[1])
        y_obs = observational_prior(
            name="yObserved", mu=y_model, sigma=epsilon, observed=observed_incidence
        )

    return model


def change_detection(
    trace_before: InferenceData,
    trace_after: InferenceData,
    test_and_metrics: Iterable[dict[str, Callable]],
    param: str | Iterable[str],
    kind: Literal["strata", "region"] | Iterable[Literal["strata", "region"]],
) -> pd.DataFrame:
    """
    Compare two traces for given strata parameters.

    Args:
        trace_before: The trace before the change.
        trace_after: The trace after the change.
        test_and_metrics: A list of dictionaries with the test and metric to apply. The
            first key is the name of the test and the second key is the name of the
            metric.
        param: The parameter(s) to compare.
        kind: The kind of parameter to compare, either "strata" or "region".

    Returns:
        A pandas DataFrame with the columns "kind", "param", "name", "test", "metric",
        "test_statistic", "p_value", and "metric_value".
    """
    test_and_metrics = (
        [test_and_metrics] if isinstance(test_and_metrics, dict) else test_and_metrics
    )
    param = [param] if isinstance(param, str) else param
    kind = [kind] if isinstance(kind, str) else kind

    for k in kind:
        if not np.all(
            trace_before.posterior.coords.get(k).to_numpy()
            == trace_after.posterior.coords.get(k).to_numpy()
        ):
            raise ValueError(
                f"The '{k}' coordinates do not match between the two traces."
            )

    results = []

    for k in kind:
        for p in param:
            for s in trace_before.posterior.coords.get(k).to_numpy().tolist():
                kwargs = {}
                kwargs[k] = s
                x = (
                    trace_after.posterior[f"{p}{k.title()}"]
                    .sel(**kwargs)
                    .to_numpy()
                    .flatten()
                )
                y = (
                    trace_before.posterior[f"{p}{k.title()}"]
                    .sel(**kwargs)
                    .to_numpy()
                    .flatten()
                )
                for test_and_metric in test_and_metrics:
                    test_name, metric_name = test_and_metric.keys()
                    test_func, metric_func = test_and_metric.values()
                    test_result = test_func(x, y)
                    metric_value = metric_func(x, y)
                    results.append(
                        {
                            "kind": k,
                            "param": p,
                            "name": s,
                            "test": test_name,
                            "metric": metric_name,
                            "test_statistic": test_result.statistic,
                            "p_value": test_result.pvalue,
                            "metric_value": metric_value,
                        }
                    )

    results = pd.DataFrame.from_records(results)

    return results
