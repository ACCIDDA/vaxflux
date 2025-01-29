__all__ = ("build_model", "change_detection", "posterior_forecast")


from collections.abc import Iterable
from typing import Any, Callable, Literal

from arviz import InferenceData
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from vaxflux._util import _clean_text
from vaxflux.curves import IncidenceCurve
from vaxflux.data import coordinates_from_incidence


def build_model(
    data: pd.DataFrame,
    observation_type: Literal["incidence", "prevalence"],
    value_type: Literal["rate", "count"],
    incidence_curve: IncidenceCurve,
    observational_dist: pm.Distribution,
    epsilon_prior: tuple[pm.Distribution, dict[str, Any]],
    parameter_priors: dict[str, tuple[pm.Distribution, dict[str, Any]]],
    season_stratified_parameters: tuple[str, ...] = (),
    season_walk_sigma: float | dict[str, float] = 0.01,
) -> pm.Model:
    """
    Build an model for vaccine incidence.

    Args:
        data: DataFrame with columns "time", "season", "region", "strata", and "value".
        observation_type: The type of observation, either "incidence" or "prevalence".
        value_type: The type of value, either "rate" or "count".
        incidence_curve: An `IncidenceCurve` to fit the uptake with.
        observational_dist: A PyMC distribution for the observational model.
        epsilon_prior: A tuple with the PyMC distribution for the epsilon parameter and
            its parameters.
        parameter_priors: A dictionary with the PyMC distribution for each parameter and
            its parameters.
        season_stratified_parameters: A tuple with the parameters that are stratified by
            season.
        season_walk_sigma: The sigma for the Gaussian random walk for strata/region
            specific parameters in `season_stratified_parameters`. Can be a float for
            all parameters or a dictionary with the sigma for each parameter.

    Returns:
        A PyMC model object.
    """
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
            _strata_region_factors(
                p,
                "region",
                params,
                coords,
                parameter_priors,
                season_stratified_parameters,
                season_walk_sigma,
            )
            # Strata
            _strata_region_factors(
                p,
                "strata",
                params,
                coords,
                parameter_priors,
                season_stratified_parameters,
                season_walk_sigma,
            )

        # **Model computations**
        evaluate_params = {
            p: pm.Deterministic(
                p,
                (
                    params[f"{p}Macro"][indexes[f"{p}Season"]]
                    + params[f"{p}Region"][indexes["region"]]
                    + (
                        pt.stack(
                            [
                                params[f"{p}Strata"][indexes["strata"][i]][
                                    indexes["season"][i]
                                ]
                                for i in range(len(data))
                            ],
                            axis=0,
                        )
                        if isinstance(params[f"{p}Strata"], list)
                        else params[f"{p}Strata"][indexes["strata"]]
                    )
                ),
                dims="observation",
            )
            for p in incidence_curve.parameters
        }
        # TODO: This is not quite the right way to model prevalence, prevalence should
        # be the integral of the incidence curve. Here errors in observed prevalence
        # are independent of time, which is not true.
        y_model = pm.Deterministic(
            "yModel",
            (
                incidence_curve.evaluate(time, **evaluate_params)
                if observation_type == "incidence"
                else incidence_curve.prevalence(time, **evaluate_params)
            ),
            dims="observation",
        )

        # **Observational model**
        epsilon = epsilon_prior[0](name="epsilon", **epsilon_prior[1])
        y_obs = observational_dist(
            name="yObserved", mu=y_model, sigma=epsilon, observed=observed_incidence
        )

    return model


def _strata_region_factors(
    p: str,
    kind: Literal["region", "strata"],
    params: dict[str, Any],
    coords: dict[Literal["season", "region", "strata", "observation"], list[str]],
    parameter_priors: dict[str, tuple[pm.Distribution, dict[str, Any]]],
    season_stratified_parameters: tuple[str, ...],
    season_walk_sigma: float | dict[str, float],
) -> None:
    """
    Helper to construct the region/strata specific factors for a parameter.

    Args:
        p: The parameter name.
        kind: The kind of factor, either "region" or "strata".
        params: The dictionary to store the PyMC variables.
        coords: The coordinates.
        parameter_priors: The parameter priors.
        season_stratified_parameters: The parameters that are stratified by season.
        season_walk_sigma: The sigma for the Gaussian random walk for strata/region
            specific parameters in `season_stratified_parameters

    Returns:
        None
    """
    if (p_kind := f"{p}{kind.title()}") in parameter_priors:
        dist, dist_params = parameter_priors[p_kind]
        if p_kind in season_stratified_parameters:
            params[p_kind] = []
            for s in coords[kind]:
                params[p_kind].append(
                    pm.GaussianRandomWalk(
                        name=f"{p_kind}{_clean_text(s)}",
                        sigma=(
                            season_walk_sigma
                            if isinstance(season_walk_sigma, float)
                            else season_walk_sigma[p_kind]
                        ),
                        init_dist=dist.dist(
                            **{
                                k: (v[s] if isinstance(v, dict) else v)
                                for k, v in dist_params.items()
                            }
                        ),
                        dims="season",
                    )
                )
        else:
            params[p_kind] = dist(name=p_kind, dims=kind, **dist_params)
    else:
        params[p_kind] = pm.Data(p_kind, np.repeat(0.0, len(coords[kind])), dims=kind)


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


def posterior_forecast(
    times,
    trace: InferenceData,
    curve: IncidenceCurve,
) -> pd.DataFrame:
    """
    Produce a posterior forecast.

    Args:
        times: The times to forecast for.
        trace: The trace to use for the forecast.
        observation_type: The type of observation.
        curve: The incidence curve to use for the forecast.

    Returns:
        A pandas DataFrame with the columns: 'chain', 'draw', 'season', 'strata',
        'time', 'value', and 'prevalence'.
    """
    chains = trace.posterior.coords["chain"].values.tolist()
    draws = trace.posterior.coords["draw"].values.tolist()
    seasons = trace.posterior.coords["season"].values.tolist()
    strata = trace.posterior.coords["strata"].values.tolist()

    times_array = np.zeros((len(times), len(chains), len(draws)))
    for i in range(len(chains)):
        for j in range(len(draws)):
            times_array[:, i, j] = times

    dfs = []
    for i, season in enumerate(seasons):
        for j, stratum in enumerate(strata):
            # shape: (chains, draws)
            m = (
                trace.posterior["mMacro"].sel(mSeason=season).values
                + trace.posterior[f"mStrata{_clean_text(stratum)}"]
                .sel(season=season)
                .values
            )
            # shape: (chains, draws)
            r = (
                trace.posterior["rMacro"].sel(rSeason=season).values
                + trace.posterior[f"rStrata"].sel(strata=stratum).values
            )
            # shape: (chains, draws)
            s = (
                trace.posterior["sMacro"].sel(sSeason=season).values
                + trace.posterior[f"sStrata"].sel(strata=stratum).values
            )
            # shape: (times, chains, draws)
            y = curve.evaluate(times_array, m, r, s).eval()
            # z = cumulative_trapezoid(y, x=times_array, initial=0.0, axis=0)
            z = curve.prevalence(times_array, m, r, s).eval()

            # Append to list of DataFrames
            # TODO: Inefficient, but it's the easiest way to do this for now
            for k, chain in enumerate(chains):
                for l, draw in enumerate(draws):
                    dfs.append(
                        pd.DataFrame(
                            {
                                "chain": chain,
                                "draw": draw,
                                "season": season,
                                "strata": stratum,
                                "time": times,
                                "value": y[:, k, l],
                                "prevalence": z[:, k, l],
                            }
                        )
                    )

    simulated = pd.concat(dfs)
    return simulated


def model_parameter_summary(
    trace: InferenceData, curve: IncidenceCurve
) -> pd.DataFrame:
    """
    Produce a summary of the fitted model parameters.

    Args:
        trace: The posterior trace to summarize.
        curve: The incidence curve used in the model.

    Returns:
        A pandas DataFrame.
    """
    raise NotImplementedError
