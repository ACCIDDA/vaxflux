"""
Functionality for creating a multilevel model of vaccine uptake.

This module contains the needed utilities to create and fit a multilevel model of
vaccine uptake.
"""

__all__ = ("UptakeModelConfig", "create_multilevel_model", "generate_model_outputs")


from dataclasses import dataclass

import arviz as az
import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc as pm
from scipy.special import expit
import xarray as xr


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
        self._coords: dict[int, dict[str, list[str]]] = {}
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
        """Return a representation of this object with optional human readable name."""
        if self.name:
            return f"<vaxflux.multilevel_model.UptakeModelConfig named '{self.name}'>"
        return f"<vaxflux.multilevel_model.UptakeModelConfig with id {id(self)}>"

    @property
    def mu_k(self) -> float:  # noqa: D102
        return self.k_prior[0]

    @property
    def sigma_k(self) -> float:  # noqa: D102
        return self.k_prior[1]

    @property
    def mu_r(self) -> float:  # noqa: D102
        return self.s_prior[0]

    @property
    def sigma_r(self) -> float:  # noqa: D102
        return self.s_prior[1]

    @property
    def mu_s(self) -> float:  # noqa: D102
        return self.s_prior[0]

    @property
    def sigma_s(self) -> float:  # noqa: D102
        return self.s_prior[1]

    @property
    def mu_dk_region(self) -> float | None:  # noqa: D102
        return self.dk_region_prior[0] if self.dk_region_prior else None

    @property
    def sigma_dk_region(self) -> float | None:  # noqa: D102
        return self.dk_region_prior[1] if self.dk_region_prior else None

    @property
    def mu_dr_region(self) -> float | None:  # noqa: D102
        return self.dr_region_prior[0] if self.dr_region_prior else None

    @property
    def sigma_dr_region(self) -> float | None:  # noqa: D102
        return self.dr_region_prior[1] if self.dr_region_prior else None

    @property
    def mu_ds_region(self) -> float | None:  # noqa: D102
        return self.ds_region_prior[0] if self.ds_region_prior else None

    @property
    def sigma_ds_region(self) -> float | None:  # noqa: D102
        return self.ds_region_prior[1] if self.ds_region_prior else None

    @property
    def mu_dk_strata(self) -> float | None:  # noqa: D102
        return self.dk_strata_prior[0] if self.dk_strata_prior else None

    @property
    def sigma_dk_strata(self) -> float | None:  # noqa: D102
        return self.dk_strata_prior[1] if self.dk_strata_prior else None

    @property
    def mu_dr_strata(self) -> float | None:  # noqa: D102
        return self.dr_strata_prior[0] if self.dr_strata_prior else None

    @property
    def sigma_dr_strata(self) -> float | None:  # noqa: D102
        return self.dr_strata_prior[1] if self.dr_strata_prior else None

    @property
    def mu_ds_strata(self) -> float | None:  # noqa: D102
        return self.ds_strata_prior[0] if self.ds_strata_prior else None

    @property
    def sigma_ds_strata(self) -> float | None:  # noqa: D102
        return self.ds_strata_prior[1] if self.ds_strata_prior else None

    @property
    def coords(self) -> dict[str, list[str]]:  # noqa: D102
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
    def coord_dims(self) -> dict[str, int]:  # noqa: D102
        return {k: len(v) for k, v in self.coords.items()}

    @property
    def season_index(self):  # noqa: D102
        return (
            self.data["season"].apply(lambda x: self.coords["season"].index(x)).values
        )

    @property
    def region_index(self):  # noqa: D102
        return (
            self.data["region"].apply(lambda x: self.coords["region"].index(x)).values
        )

    @property
    def strata_index(self):  # noqa: D102
        return (
            self.data["strata"].apply(lambda x: self.coords["strata"].index(x)).values
        )

    @property
    def k_season_index(self):  # noqa: D102
        return self.season_index if self.k_season_stratified else np.array([0])

    @property
    def r_season_index(self):  # noqa: D102
        return self.season_index if self.r_season_stratified else np.array([0])

    @property
    def s_season_index(self):  # noqa: D102
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
        # TODO: Should the error term in this observational model be moved to inside
        # the logistic? That way errors stay bounded to withing plausible, this model
        # could produce estimates below 0 for vaccination rate.
        eps = pm.HalfNormal("eps", sigma=config.eps_prior)
        pm.Normal("y_obs", mu=y_model, sigma=eps, observed=rate)

    # Return
    return model


def generate_model_outputs(
    trace: pm.backends.base.MultiTrace | az.InferenceData,
    t: npt.NDArray[np.number],
) -> xr.DataArray:
    """
    Generate a distribution of model outputs for a multilevel model.

    Args:
        trace: A pymc trace object extracted from the sample model generated by
            `create_multilevel_model`.
        t: A numpy array of the time steps to calculate the output for.

    Returns:
        A numpy array with the dimensions of 'time', 'sample', 'season', 'region', and
        'strata'.

    """
    # Formatting and extraction
    if isinstance(trace, pm.backends.base.MultiTrace):
        raise NotImplementedError
    stacked = az.extract(getattr(trace, "posterior"))

    # Determine the output shape and helpers
    seasons_shape = max([stacked.sizes[f"{x}_season"] for x in ["k", "r", "s"]])
    output_shape = (
        len(t),
        stacked.sizes["sample"],
        seasons_shape,
        stacked.sizes.get("region", 1),
        stacked.sizes.get("strata", 1),
    )
    k_idx = (
        np.repeat(0, output_shape[2])
        if stacked.sizes["k_season"] == 1
        else np.arange(output_shape[2])
    )
    r_idx = (
        np.repeat(0, output_shape[2])
        if stacked.sizes["r_season"] == 1
        else np.arange(output_shape[2])
    )
    s_idx = (
        np.repeat(0, output_shape[2])
        if stacked.sizes["s_season"] == 1
        else np.arange(output_shape[2])
    )

    # Loop over non-sample dim, sample dim is usually the biggest.
    output = xr.DataArray(
        dims=["t", "sample", "season", "region", "strata"],
        coords={
            "t": t,
            "sample": stacked.coords["sample"],
            "season": next(
                stacked.coords[f"{p}_season"].values
                for p in ("k", "r", "s")
                if stacked.sizes[f"{p}_season"] == seasons_shape
            ),
            "region": stacked.coords.get("region", ["All Region"]),
            "strata": stacked.coords.get("strata", ["All Strata"]),
        },
    )
    for time_idx in range(output_shape[0]):  # time
        for season_idx in range(output_shape[2]):  # season
            for region_idx in range(output_shape[3]):  # region
                for strata_idx in range(output_shape[4]):  # strata
                    K = expit(
                        stacked.k.values[k_idx[season_idx], :]
                        + (
                            stacked.variables["dk_region"].values[region_idx, :]
                            if "dk_region" in stacked.variables
                            else 0.0
                        )
                        + (
                            stacked.variables["dk_strata"].values[strata_idx, :]
                            if "dk_strata" in stacked.variables
                            else 0.0
                        )
                    )
                    R = (
                        stacked.r.values[r_idx[season_idx], :]
                        + (
                            stacked.variables["dr_region"].values[region_idx, :]
                            if "dr_region" in stacked.variables
                            else 0.0
                        )
                        + (
                            stacked.variables["dr_strata"].values[strata_idx, :]
                            if "dr_strata" in stacked.variables
                            else 0.0
                        )
                    )
                    S = (
                        stacked.s.values[s_idx[season_idx], :]
                        + (
                            stacked.variables["ds_region"].values[region_idx, :]
                            if "ds_region" in stacked.variables
                            else 0.0
                        )
                        + (
                            stacked.variables["ds_strata"].values[strata_idx, :]
                            if "ds_strata" in stacked.variables
                            else 0.0
                        )
                    )
                    output[time_idx, :, season_idx, region_idx, strata_idx] = K * expit(
                        R * (t[time_idx] - S)
                    )

    return output
