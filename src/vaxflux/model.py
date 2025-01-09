__all__ = ("build_model",)


from typing import Any, Literal

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
        {f"{p}_season": coords["season"] for p in season_stratified_parameters}
    )
    coords.update(
        {
            f"{p}_season": ["All Seasons"]
            for p in set(incidence_curve.parameters) - set(season_stratified_parameters)
        }
    )
    indexes = {
        column: data[column].apply(lambda x: coords[column].index(x)).values
        for column in ("season", "region", "strata")
    }
    indexes.update(
        {f"{p}_season": indexes["season"] for p in season_stratified_parameters}
    )
    indexes.update(
        {
            f"{p}_season": np.array([0])
            for p in set(incidence_curve.parameters) - set(season_stratified_parameters)
        }
    )

    # Build the model
    with pm.Model(coords=coords) as model:
        # **Data**
        time = pm.Data("time", data["time"].values)
        observed_incidence = pm.Data("observed_incidence", data["incidence"].values)

        # **Prior distributions**
        params = {}
        for p in incidence_curve.parameters:
            # Macro
            p_season = f"{p}_macro"
            dist, dist_params = parameter_priors[p]
            params[p_season] = dist(name=p_season, dims=f"{p}_season", **dist_params)
            # Region
            if (p_region := f"{p}_region") in parameter_priors:
                dist, dist_params = parameter_priors[p_region]
                params[p_region] = dist(name=p_region, dims="region", **dist_params)
            else:
                params[p_region] = pm.Data(
                    p_region, np.repeat(0.0, len(coords["region"])), dims="region"
                )
            # Strata
            if (p_strata := f"{p}_strata") in parameter_priors:
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
                params[f"{p}_macro"][indexes[f"{p}_season"]]
                + params[f"{p}_region"][indexes["region"]]
                + params[f"{p}_strata"][indexes["strata"]],
                dims="observation",
            )
            for p in incidence_curve.parameters
        }
        y_model = pm.Deterministic(
            "y_model",
            incidence_curve.evaluate(time, **evaluate_params),
            dims="observation",
        )

        # **Observational model**
        epsilon = epsilon_prior[0](name="epsilon", **epsilon_prior[1])
        y_obs = observational_prior(
            name="y_obs", mu=y_model, sigma=epsilon, observed=observed_incidence
        )

    return model
