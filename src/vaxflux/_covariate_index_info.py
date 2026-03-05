from typing import NamedTuple

from vaxflux._covariates import Covariate


class CovariateIndexInfo(NamedTuple):
    """Pre-resolved indexing data for a single covariate, built during `_pre_model`.

    Attributes:
        covariate: The covariate object itself.
        resolved_categories: The resolved `list[str]` of categories for categorical
            covariates, or `None` for seasonal covariates.
        combo_position: For categorical covariates, the position of this covariate's
            name in the category-combo tuple (used to extract the right category value
            from each combo). `None` for seasonal covariates.
        category_index_map: For categorical covariates, a mapping from category string
            to its integer index into the sampled values array. `None` for seasonal
            covariates.
    """

    covariate: Covariate
    resolved_categories: list[str] | None
    combo_position: int | None
    category_index_map: dict[str, int] | None


__all__ = []
