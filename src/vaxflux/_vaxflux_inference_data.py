from functools import cached_property
from itertools import product
from typing import Any, cast

import arviz as az
import pandas as pd
import xarray as xr

from vaxflux._util import _coord_name


class VaxfluxInferenceData(az.InferenceData):
    """
    Container for inference data specific to vaxflux.

    This class extends
    [ArviZ's `InferenceData`](https://python.arviz.org/en/stable/api/generated/arviz.InferenceData.html)
    to include specialized functionality for vaxflux models. This allows for users to
    easily use this object with functions that expect ArviZ `InferenceData` while also
    providing vaxflux-specific methods and attributes.

    """

    @classmethod
    def from_numpyro(cls, *args: Any, **kwargs: Any) -> "VaxfluxInferenceData":
        """
        Create inference data from NumPyro outputs via ArviZ.

        For more details, see
        [ArviZ's `from_numpyro` function](https://python.arviz.org/en/stable/api/generated/arviz.from_numpyro.html).

        Args:
            *args: Positional arguments forwarded to `arviz.from_numpyro`.
            **kwargs: Keyword arguments forwarded to `arviz.from_numpyro`.

        Returns:
            A `VaxfluxInferenceData` instance populated from NumPyro results.

        """
        idata = cast("az.InferenceData", az.from_numpyro(*args, **kwargs))  # type: ignore[no-untyped-call]
        idata.__class__ = cls
        return cast("VaxfluxInferenceData", idata)

    @cached_property
    def merged_prior(self) -> xr.Dataset:
        """
        Return a merged prior dataset combining prior and prior predictive.

        Returns:
            An xarray Dataset with prior and prior predictive variables.

        """
        prior = getattr(self, "prior", None)
        prior_predictive = getattr(self, "prior_predictive", None)
        if prior is None and prior_predictive is None:
            msg = "Prior predictive data not available in this inference object."
            raise ValueError(msg)
        datasets = [ds for ds in (prior, prior_predictive) if ds is not None]
        return cast("xr.Dataset", xr.merge(datasets, compat="no_conflicts"))

    @cached_property
    def merged_posterior(self) -> xr.Dataset:
        """
        Return a merged posterior dataset combining posterior and posterior predictive.

        Returns:
            An xarray Dataset with posterior and posterior predictive variables.

        """
        posterior = getattr(self, "posterior", None)
        posterior_predictive = getattr(self, "posterior_predictive", None)
        if posterior is None and posterior_predictive is None:
            msg = "Posterior predictive data not available in this inference object."
            raise ValueError(msg)
        datasets = [ds for ds in (posterior, posterior_predictive) if ds is not None]
        return cast("xr.Dataset", xr.merge(datasets, compat="no_conflicts"))

    @cached_property
    def coords(self) -> dict[str, list[str]]:
        """
        Return model coordinates without chain/draw dimensions.

        Returns:
            The coordinate mapping for the merged prior or posterior dataset without
            chain and draw dimensions (which differ between prior/posterior).

        """
        if (
            getattr(self, "prior", None) is not None
            or getattr(self, "prior_predictive", None) is not None
        ):
            dataset = self.merged_prior
        elif (
            getattr(self, "posterior", None) is not None
            or getattr(self, "posterior_predictive", None) is not None
        ):
            dataset = self.merged_posterior
        else:
            msg = "No data available in this inference object."
            raise ValueError(msg)
        coords = {}
        for name in dataset.coords:
            if name in {"chain", "draw"}:
                continue
            coords[str(name)] = cast(
                "list[str]", dataset.coords[name].to_numpy().tolist()
            )
        return coords

    @cached_property
    def covariate_categories(self) -> dict[str, dict[str, str]]:
        """
        A mapping of covariate names to their category labels.

        Returns:
            A dictionary mapping covariate names to dictionaries of category labels
            where each inner dictionary maps coordinate-safe category names to the
            original category labels.

        """
        covariate_names = {
            key[: -len("_categories")]
            for key in self.coords
            if key.endswith("_categories") and not key.endswith("_categories_short")
        }
        covariate_category_map: dict[str, dict[str, str]] = {}
        for covariate_name in covariate_names:
            coord_key = f"{covariate_name}_categories"
            if coord_key not in self.coords:
                continue
            categories = self.coords[coord_key]
            covariate_category_map[covariate_name] = {
                _coord_name(str(cat)): str(cat) for cat in categories
            }
        return covariate_category_map

    @cached_property
    def prior_observations(self) -> pd.DataFrame:
        """
        Return prior observations as a formatted DataFrame.

        Returns:
            A DataFrame with observation metadata and values. Will contain the columns
            'chain', 'draw', 'season', 'season_start_date', 'season_end_date',
            'start_date', 'end_date', 'report_date', any covariate columns,
            'type', and 'value'.

        """
        return self._observations_from_dataset(self.merged_prior)

    @cached_property
    def posterior_observations(self) -> pd.DataFrame:
        """
        Return posterior observations as a formatted DataFrame.

        Returns:
            A DataFrame with observation metadata and values. Will contain the columns
            'chain', 'draw', 'season', 'season_start_date', 'season_end_date',
            'start_date', 'end_date', 'report_date', any covariate columns,
            'type', and 'value'.

        """
        return self._observations_from_dataset(self.merged_posterior)

    def _observations_from_dataset(self, dataset: xr.Dataset) -> pd.DataFrame:
        """
        Build an observations DataFrame from an ArviZ dataset.

        This extracts incidence curve parameters and reconstructs observation metadata
        from stored coordinates and variable naming conventions.

        """
        observations: list[pd.DataFrame] = []

        # Find incidence variables to decode into observation rows
        incidence_vars = [
            name for name in dataset.data_vars if str(name).startswith("incidence_")
        ]
        if not incidence_vars:
            msg = "No incidence variables found in the inference data."
            raise ValueError(msg)

        covariate_names = self.coords.get("covariate_names", [])
        if not covariate_names:
            covariate_names = sorted(
                key[: -len("_categories")]
                for key in self.coords
                if key.endswith("_categories") and not key.endswith("_categories_short")
            )
        covariate_categories = {
            name: self.coords.get(f"{name}_categories", []) for name in covariate_names
        }
        season_labels = self.coords.get("season", [])
        date_dims = [dim for dim in dataset.dims if str(dim).startswith("date_ranges_")]

        # Convert each season/category combination into observation rows
        for date_dim in date_dims:
            season_token = str(date_dim)[len("date_ranges_") :]
            season = next(
                (
                    season_name
                    for season_name in season_labels
                    if _coord_name(season_name) == season_token
                ),
                season_token,
            )
            category_combos = (
                list(product(*(covariate_categories[name] for name in covariate_names)))
                if covariate_names
                else [()]
            )
            for combo in category_combos:
                name_parts = ["incidence", season]
                for covariate_name, category_value in zip(
                    covariate_names, combo, strict=False
                ):
                    name_parts.extend([covariate_name, category_value])
                var_name = _coord_name(*name_parts)
                if var_name not in dataset.data_vars:
                    continue
                data = dataset[var_name]
                df = data.to_dataframe(name="value").reset_index()

                # Parse start/end/report dates from the date-range coordinate
                date_values = df[date_dim].str.split("_", n=1, expand=True)
                df["start_date"] = pd.to_datetime(date_values[0])
                df["end_date"] = pd.to_datetime(date_values[1])
                df["report_date"] = df["end_date"]
                df["season"] = season

                # Fill season start/end dates from the season day coordinates if present
                days_key = _coord_name("days", season)
                if days_key in dataset.coords:
                    days = dataset.coords[days_key].to_numpy().tolist()
                    df["season_start_date"] = (
                        pd.to_datetime(days[0]) if days else pd.NaT
                    )
                    df["season_end_date"] = pd.to_datetime(days[-1]) if days else pd.NaT
                else:
                    df["season_start_date"] = pd.NaT
                    df["season_end_date"] = pd.NaT

                for covariate_name, category_value in zip(
                    covariate_names, combo, strict=False
                ):
                    df[covariate_name] = category_value

                # Finalize this block of observations and move to the next
                df["type"] = "incidence"
                df = df.drop(columns=[date_dim])
                observations.append(df)

        if not observations:
            msg = "No observation data could be constructed from incidence outputs."
            raise ValueError(msg)

        df = pd.concat(observations, ignore_index=True)
        return df[
            [
                "chain",
                "draw",
                "season",
                "season_start_date",
                "season_end_date",
                "start_date",
                "end_date",
                "report_date",
                *covariate_names,
                "type",
                "value",
            ]
        ]
