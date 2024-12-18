__all__ = ("build_model",)


from typing import Any

import numpy as np
import pandas as pd
import pymc as pm

from vaxflux.curves import IncidenceCurve
from vaxflux.data import coordinates_from_incidence


def build_model(
    incidence: pd.DataFrame,
    incidence_curve: IncidenceCurve,
    observational_prior: pm.Distribution,
    epsilon_prior: tuple[pm.Distribution, dict[str, Any]],
    parameter_priors: dict[str, tuple[pm.Distribution, dict[str, Any]]],
    season_stratified_parameters: tuple[str] = (),
) -> pm.Model:
    """
    Build an model for vaccine incidence.

    Args:
        ...
    """
    # Determining the coordinates and parameters
    coords = coordinates_from_incidence(incidence)
    coords.update(
        {f"{p}_season": coords["season"] for p in season_stratified_parameters}
    )
    coords.update(
        {
            f"{p}_season": ["All Seasons"]
            for p in set(incidence_curve.parameters)
            - set(season_stratified_parameters)
        }
    )
    indexes = {
        column: incidence[column].apply(lambda x: coords[column].index(x)).values
        for column in ("season", "region", "strata")
    }
    indexes.update(
        {f"{p}_season": indexes["season"] for p in season_stratified_parameters}
    )
    indexes.update(
        {
            f"{p}_season": np.array([0])
            for p in set(incidence_curve.parameters)
            - set(season_stratified_parameters)
        }
    )

    # Build the model
    with pm.Model(coords=coords) as model:
        # **Data**
        time = pm.Data("time", incidence["time"].values)
        observed_incidence = pm.Data(
            "observed_incidence", incidence["incidence"].values
        )

        # **Prior distributions**
        params = {}
        for p in incidence_curve.parameters:
            # Macro
            p_season = f"{p}_macro"
            # params[p_season] = parameter_priors[p].pymc_distribution(
            #     name=p_season, dims=f"{p}_season"
            # )
            dist, dist_params = parameter_priors[p]
            params[p_season] = dist(name=p_season, dims=f"{p}_season", **dist_params)
            # Region
            if (p_region := f"{p}_region") in parameter_priors:
                params[p_region] = parameter_priors[p_region].pymc_distribution(
                    name=p_region, dims="region"
                )
            else:
                params[p_region] = pm.Data(
                    p_region, np.repeat(0.0, len(coords["region"])), dims="region"
                )
            # Strata
            if (p_strata := f"{p}_strata") in parameter_priors:
                params[p_strata] = parameter_priors[p_strata].pymc_distribution(
                    name=p_strata, dims="strata"
                )
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
        # epsilon = epsilon_prior.pymc_distribution(name="epsilon")
        epsilon = epsilon_prior[0](name="epsilon", **epsilon_prior[1])
        y_obs = observational_prior(
            name="y_obs", mu=y_model, sigma=epsilon, observed=observed_incidence
        )

    return model
