"""
Functionality for creating a multilevel model of vaccine uptake.

This module contains the needed utilities to create and fit a multilevel model of 
vaccine uptake.
"""

__all__ = ["UptakeModelConfig", "create_multilevel_model"]


from dataclasses import dataclass

import numpy as np
import pandas as pd
import pymc as pm


@dataclass
class UptakeModelConfig:
    """
    A vaccine uptake model config.

    This immutable class provides the configuration for describing and constructing a
    vaccine uptake model.

    Attributes:
        name: An optional name to give to this model config. Only used for pretty
            printing.
        TODO: Document the attributes of this config class
    """

    data: pd.DataFrame

    eps_prior: float

    k_prior: tuple[float, float]
    r_prior: tuple[float, float]
    s_prior: tuple[float, float]

    dk_region_prior: tuple[float, float] | None = None
    dr_region_prior: tuple[float, float] | None = None
    ds_region_prior: tuple[float, float] | None = None

    dk_strata_prior: tuple[float, float] | None = None
    dr_strata_prior: tuple[float, float] | None = None
    ds_strata_prior: tuple[float, float] | None = None

    k_season_stratified: bool = True
    r_season_stratified: bool = False
    s_season_stratified: bool = False

    name: str | None = None

    def __post_init__(self) -> None:
        """
        Post initialization hook for input validation.

        Raises:
            ValueError: If `data` does not have all of the required columns: 'time',
                'season', 'region', 'strata', and 'rate'.
        """
        # Private instance attributes
        self._coords = {}
        # First create local copy as to not clobber user variable and then validate
        self.data = self.data.copy()
        required_columns = {"time", "season", "region", "strata", "rate"}
        if missing_columns := required_columns - set(self.data.columns.tolist()):
            raise ValueError(
                (
                    "The `data` provided is missing required columns: "
                    f"""'{"', '".join(missing_columns)}'."""
                )
            )
        self.data = self.data[list(required_columns)]
        # TODO: Check that there are no NA values in `data`
        # Coerce columns of `data` to the right types and reorder
        self.data["time"] = pd.to_numeric(self.data["time"]).astype("float64")
        self.data["season"] = self.data["season"].astype("string")
        self.data["region"] = self.data["region"].astype("string")
        self.data["strata"] = self.data["strata"].astype("string")
        self.data["rate"] = pd.to_numeric(self.data["rate"]).astype("float64")
        self.data = self.data[["season", "region", "strata", "time", "rate"]]
        # self.data = self.data.set_index(["season", "region", "strata"])

    def __repr__(self) -> str:
        if self.name:
            return f"<vaxflux.multilevel_model.UptakeModelConfig named '{self.name}'>"
        return f"<vaxflux.multilevel_model.UptakeModelConfig with id {id(self)}>"

    @property
    def mu_k(self) -> float:
        return self.k_prior[0]

    @property
    def sigma_k(self) -> float:
        return self.k_prior[1]

    @property
    def mu_r(self) -> float:
        return self.s_prior[0]

    @property
    def sigma_r(self) -> float:
        return self.s_prior[1]

    @property
    def mu_s(self) -> float:
        return self.s_prior[0]

    @property
    def sigma_s(self) -> float:
        return self.s_prior[1]

    @property
    def mu_dk_region(self) -> float | None:
        return self.dk_region_prior[0] if self.dk_region_prior else None

    @property
    def sigma_dk_region(self) -> float | None:
        return self.dk_region_prior[1] if self.dk_region_prior else None

    @property
    def mu_dr_region(self) -> float | None:
        return self.dr_region_prior[0] if self.dr_region_prior else None

    @property
    def sigma_dr_region(self) -> float | None:
        return self.dr_region_prior[1] if self.dr_region_prior else None

    @property
    def mu_ds_region(self) -> float | None:
        return self.ds_region_prior[0] if self.ds_region_prior else None

    @property
    def sigma_ds_region(self) -> float | None:
        return self.ds_region_prior[1] if self.ds_region_prior else None

    @property
    def mu_dk_strata(self) -> float | None:
        return self.dk_strata_prior[0] if self.dk_strata_prior else None

    @property
    def sigma_dk_strata(self) -> float | None:
        return self.dk_strata_prior[1] if self.dk_strata_prior else None

    @property
    def mu_dr_strata(self) -> float | None:
        return self.dr_strata_prior[0] if self.dr_strata_prior else None

    @property
    def sigma_dr_strata(self) -> float | None:
        return self.dr_strata_prior[1] if self.dr_strata_prior else None

    @property
    def mu_ds_strata(self) -> float | None:
        return self.ds_strata_prior[0] if self.ds_strata_prior else None

    @property
    def sigma_ds_strata(self) -> float | None:
        return self.ds_strata_prior[1] if self.ds_strata_prior else None

    @property
    def coords(self) -> dict[str, list[str]]:
        if (coords := self._coords.get(id(self.data))) is None:
            coords = {
                v: self.data[v].unique().tolist()
                for v in ["season", "region", "strata"]
            }
            for param, season_stratified in zip(
                ["k", "r", "s"],
                [
                    self.k_season_stratified,
                    self.r_season_stratified,
                    self.s_season_stratified,
                ],
            ):
                coords[f"{param}_season"] = (
                    coords["season"] if season_stratified else ["All Seasons"]
                )
            self._coords[id(self.data)] = coords
        return coords.copy()

    @property
    def coord_dims(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.coords.items()}

    @property
    def season_index(self):
        return (
            self.data["season"].apply(lambda x: self.coords["season"].index(x)).values
        )

    @property
    def region_index(self):
        return (
            self.data["region"].apply(lambda x: self.coords["region"].index(x)).values
        )

    @property
    def strata_index(self):
        return (
            self.data["strata"].apply(lambda x: self.coords["strata"].index(x)).values
        )

    @property
    def k_season_index(self):
        return self.season_index if self.k_season_stratified else np.array([0])

    @property
    def r_season_index(self):
        return self.season_index if self.r_season_stratified else np.array([0])

    @property
    def s_season_index(self):
        return self.season_index if self.s_season_stratified else np.array([0])


def create_multilevel_model(config: UptakeModelConfig) -> pm.Model:
    """
    Construct a PyMC regional multilevel model for vaccine uptake.

    This function will construct a PyMC regional multilevel model where baseline
    parameters are drawn from a season prior and then regional deltas are drawn from a
    region prior. These season and region model parameters are then combined to build
    a model that fits vaccine uptake across multiple seasons/regions simultaneously.

    Args:
        config: An uptake model config object describing the model to construct.

    Returns:
        A PyMC model generated from the given uptake model configuration.
    """
    # Local inputs
    t = config.data["time"].values.copy()
    rate = config.data["rate"].values.copy()

    # Construct the model
    with pm.Model(coords=config.coords) as model:
        # **Prior distributions**
        # Macro priors
        k = pm.Normal("k", mu=config.mu_k, sigma=config.sigma_k, dims="k_season")
        r = pm.Normal("r", mu=config.mu_r, sigma=config.sigma_r, dims="r_season")
        s = pm.Normal("s", mu=config.mu_s, sigma=config.sigma_s, dims="s_season")
        # Regional priors
        if config.dk_region_prior:
            dk_region = pm.Normal(
                "dk_region",
                mu=config.mu_dk_region,
                sigma=config.sigma_dk_region,
                dims="region",
            )
        else:
            dk_region = pm.Data(
                "dk_region", np.repeat(0.0, config.coord_dims["region"]), dims="region"
            )
        if config.dr_region_prior:
            dr_region = pm.Normal(
                "dr_region",
                mu=config.mu_dr_region,
                sigma=config.sigma_dr_region,
                dims="region",
            )
        else:
            dr_region = pm.Data(
                "dr_region", np.repeat(0.0, config.coord_dims["region"]), dims="region"
            )
        if config.ds_region_prior:
            ds_region = pm.Normal(
                "ds_region",
                mu=config.mu_ds_region,
                sigma=config.sigma_ds_region,
                dims="region",
            )
        else:
            ds_region = pm.Data(
                "ds_region", np.repeat(0.0, config.coord_dims["region"]), dims="region"
            )
        # Strata priors
        if config.dk_strata_prior:
            dk_strata = pm.Normal(
                "dk_strata",
                mu=config.mu_dk_strata,
                sigma=config.sigma_dk_strata,
                dims="strata",
            )
        else:
            dk_strata = pm.Data(
                "dk_strata", np.repeat(0.0, config.coord_dims["strata"]), dims="strata"
            )
        if config.dr_strata_prior:
            dr_strata = pm.Normal(
                "dr_strata",
                mu=config.mu_dr_strata,
                sigma=config.sigma_dr_strata,
                dims="strata",
            )
        else:
            dr_strata = pm.Data(
                "dr_strata", np.repeat(0.0, config.coord_dims["strata"]), dims="strata"
            )
        if config.ds_strata_prior:
            ds_strata = pm.Normal(
                "ds_strata",
                mu=config.mu_ds_strata,
                sigma=config.sigma_ds_strata,
                dims="strata",
            )
        else:
            ds_strata = pm.Data(
                "ds_strata", np.repeat(0.0, config.coord_dims["strata"]), dims="strata"
            )

        # **Model computations**
        # Calculate K, R, S intermediates
        K = pm.math.invlogit(
            k[config.k_season_index]
            + dk_region[config.region_index]
            + dk_strata[config.strata_index]
        )
        R = (
            r[config.r_season_index]
            + dr_region[config.region_index]
            + dr_strata[config.strata_index]
        )
        S = (
            s[config.s_season_index]
            + ds_region[config.region_index]
            + ds_strata[config.strata_index]
        )

        # Calculate model curve
        y_model = K * pm.math.invlogit(R * (t - S))

        # **Observational model**
        # TODO: Should the error term in this observational model be moved to inside the
        # logistic? That way errors stay bounded to withing plausible, this model could
        # produce estimates below 0 for vaccination rate.
        eps = pm.HalfNormal("eps", sigma=config.eps_prior)
        pm.Normal("y_obs", mu=y_model, sigma=eps, observed=rate)

    # Return
    return model
