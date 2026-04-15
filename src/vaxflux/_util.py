__all__: tuple[str, ...] = ()


import re
from collections.abc import Callable, Sequence
from typing import Annotated, Any, Final, TypeVar, overload

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BeforeValidator

from vaxflux._types import NumericalArrayLike

_CLEAN_TEXT_REGEX: Final = re.compile(r"[^a-zA-Z0-9]")

T = TypeVar("T")


def _collect_args(
    args: tuple[T | Sequence[T], ...],
    expected_type: type[T],
    type_name: str,
) -> list[T]:
    """Collect and validate objects from arguments.

    Args:
        args: Variable arguments that can be objects or sequences of objects.
        expected_type: The expected type of the objects.
        type_name: Name of the type for error messages.

    Returns:
        List of validated objects.

    Raises:
        TypeError: If arguments are not of the expected type.

    Examples:
        >>> from vaxflux._util import _collect_args
        >>> _collect_args((1, 2, 3), int, "int")
        [1, 2, 3]
        >>> _collect_args(([1, 2], 3), int, "int")
        [1, 2, 3]
        >>> _collect_args((1, [2, 3], 4), int, "int")
        [1, 2, 3, 4]
        >>> _collect_args(("not an int",), int, "int")
        Traceback (most recent call last):
            ...
        TypeError: Arguments must be int objects or sequences of int objects, got str.
    """
    items: list[T] = []
    for arg in args:
        if isinstance(arg, expected_type):
            items.append(arg)
        elif isinstance(arg, Sequence) and not isinstance(arg, (str, bytes)):
            # Validate that all items in the sequence are of expected type
            for item in arg:
                if not isinstance(item, expected_type):
                    msg = (
                        f"All items in a sequence must be {type_name} objects, "
                        f"got {type(item).__name__}."
                    )
                    raise TypeError(msg)
            items.extend(arg)
        else:
            msg = (
                f"Arguments must be {type_name} objects or sequences of "
                f"{type_name} objects, got {type(arg).__name__}."
            )
            raise TypeError(msg)
    return items


def _clean_name(
    *args: str | None,
    joiner: str = "",
    transform: Callable[[str], str] = lambda x: x,
) -> str:
    """
    Generic function to clean and join names.

    Args:
        *args: The names to clean and join, `None` values are ignored.
        joiner: The string to join the cleaned names with.
        transform: The function to transform the cleaned names.

    Returns:
        A cleaned and joined name.

    Examples:
        >>> from vaxflux._util import _clean_name
        >>> _clean_name("Abc", "Def", "GHI")
        'AbcDefGHI'
        >>> _clean_name("Abc", "Def", "GHI", joiner="_")
        'Abc_Def_GHI'
        >>> _clean_name("Abc", "Def", "GHI", joiner="")
        'AbcDefGHI'
        >>> _clean_name(
        ...     "Abc", "Def", "GHI", joiner="_", transform=lambda x: x.lower()
        ... )
        'abc_def_ghi'
        >>> _clean_name(
        ...     "Abc", "Def", "GHI", joiner="_", transform=lambda x: x.upper()
        ... )
        'ABC_DEF_GHI'
        >>> _clean_name(
        ...     "Abc", "Def", "GHI", joiner="_", transform=lambda x: x.title()
        ... )
        'Abc_Def_Ghi'
        >>> _clean_name("a$b", "c#d", "e99", joiner="_")
        'a_b_c_d_e99'
    """
    return joiner.join(
        (
            transform(_CLEAN_TEXT_REGEX.sub(" ", x).strip()).replace(
                " ",
                joiner,
            )
            for x in filter(None, args)
        ),
    )


def _pm_name(*args: str | None) -> str:
    """
    Create a PyMC3 variable name from the arguments.

    The convention for PyMC3 variable names is to use title case with no spaces.

    Args:
        *args: The names to clean and join, `None` values are ignored.

    Returns:
        The cleaned and joined name.

    Examples:
        >>> from vaxflux._util import _pm_name
        >>> _pm_name("m", "age", "18-45yr")
        'MAge1845Yr'
        >>> _pm_name("nu", "county", "wake")
        'NuCountyWake'
    """
    return _clean_name(*args, transform=lambda x: x.title())


def _coord_name(*args: str | None) -> str:
    """
    Create a PyMC3 coordinate name from the arguments.

    The convention for PyMC3 coordinate names is to use lower case with underscores.

    Args:
        *args: The names to clean and join, `None` values are ignored.

    Returns:
        The cleaned and joined name.

    Examples:
        >>> from vaxflux._util import _coord_name
        >>> _coord_name("season", "2020/21", "dates")
        'season_2020_21_dates'
        >>> _coord_name("covariate", "age", "categories")
        'covariate_age_categories'
    """
    return _clean_name(*args, joiner="_", transform=lambda x: x.lower())


@overload
def _make_float_list(x: int) -> list[float]: ...


@overload
def _make_float_list(x: float) -> list[float]: ...


@overload
def _make_float_list(x: Any) -> Any: ...  # noqa: ANN401


def _make_float_list(x: float | Any) -> list[float] | Any:
    """
    Utility function to make a float list from a single float or integer.

    Args:
        x: The value to convert to a float list if an integer or float.

    Returns:
        The float list or the original value.

    Examples:
        >>> from vaxflux._util import _make_float_list
        >>> _make_float_list(1.2)
        [1.2]
        >>> _make_float_list(3)
        [3.0]
        >>> _make_float_list([1.2, 3])
        [1.2, 3]
        >>> _make_float_list((1.2, 3))
        (1.2, 3)
        >>> _make_float_list("abc")
        'abc'
        >>> _make_float_list(None) is None
        True
    """
    return [float(x)] if isinstance(x, int | float) else x


ListOfFloats = Annotated[list[float], BeforeValidator(_make_float_list)]


def _numerical_array_like_to_1d_jax_array(x: NumericalArrayLike) -> jax.Array:
    """
    Convert a numerical array-like input to a one-dimensional JAX array.

    Args:
        x: A numerical array-like input (e.g., list, tuple, NumPy array).

    Returns:
        A one-dimensional JAX array.

    Raises:
        ValueError: If the input when converted to a JAX array is not one-dimensional.

    Examples:
        >>> from vaxflux._util import _numerical_array_like_to_1d_jax_array
        >>> _numerical_array_like_to_1d_jax_array([1, 2, 3])
        Array([1., 2., 3.], dtype=float32)
        >>> _numerical_array_like_to_1d_jax_array([])
        Array([], shape=(0,), dtype=float32)
        >>> _numerical_array_like_to_1d_jax_array(1.5)
        Traceback (most recent call last):
            ...
        ValueError: Input must be a one-dimensional array, but is instead 0.
        >>> _numerical_array_like_to_1d_jax_array([[1, 2], [3, 4]])
        Traceback (most recent call last):
            ...
        ValueError: Input must be a one-dimensional array, but is instead 2.
    """
    x_array = jnp.asarray(x)
    if (ndim := x_array.ndim) != 1:
        msg = f"Input must be a one-dimensional array, but is instead {ndim}."
        raise ValueError(msg)
    return (
        x_array
        if jnp.issubdtype(x_array.dtype, jnp.inexact)
        else x_array.astype(jnp.float32)
    )


def _coord_index_dim(
    dim: str,
    season: str,
    covariate_name: str | None,
    category: str | None,
    coords: dict[str, list[str]],
) -> int:
    """
    Find the index of the requested dimension.

    Args:
        dim: The dimension to find the index of.
        season: The season for this index.
        covariate_name: The name of the covariate.
        category: The category of the covariate.
        coords: The coordinates of the model.

    Returns:
        The index of the requested dimension.

    Examples:
        >>> from vaxflux._util import _coord_index_dim
        >>> coords = {
        ...     "covariate_age_categories": ["youth", "adult", "senior"],
        ...     "covariate_age_categories_limited": ["adult", "senior"],
        ...     "covariate_names": ["sex", "age"],
        ...     "covariate_sex_categories": ["female", "male"],
        ...     "covariate_sex_categories_limited": ["male"],
        ...     "season": ["2022/2023", "2023/2024"],
        ... }
        >>> _coord_index_dim("season", "2023/2024", "sex", "male", coords)
        1
        >>> _coord_index_dim(
        ...     "covariate_sex_categories", "2023/2024", "sex", "male", coords
        ... )
        1
        >>> _coord_index_dim(
        ...     "covariate_sex_categories_limited",
        ...     "2023/2024",
        ...     "sex",
        ...     "male",
        ...     coords,
        ... )
        0
        >>> _coord_index_dim("covariate_names", "2023/2024", "sex", "male", coords)
        0
        >>> try:
        ...     _coord_index_dim(
        ...         "covariate_sex_categories_limited",
        ...         "2023/2024",
        ...         "sex",
        ...         "female",
        ...         coords,
        ...     )
        ... except Exception as e:
        ...     print(e)
        'female' is not in list

    Raises:
        NotImplementedError: If the `dim` given is unknown.
    """
    if dim == "season":
        return coords[dim].index(season)
    if (
        covariate_name is not None
        and category is not None
        and dim
        in {
            _coord_name("covariate", covariate_name, "categories"),
            _coord_name("covariate", covariate_name, "categories", "limited"),
        }
    ):
        return coords[dim].index(category)
    if covariate_name is not None and dim == "covariate_names":
        return coords[dim].index(covariate_name)
    msg = f"Unknown dimension: '{dim}'."
    raise NotImplementedError(msg)


def _coord_index(
    dims: tuple[str, ...],
    season: str,
    covariate_name: str | None,
    category: str | None,
    coords: dict[str, list[str]],
) -> tuple[int, ...] | None:
    """
    Determine the index of the RV to select.

    Args:
        dims: The dimensions of the RV.
        season: The season for this index.
        covariate_name: The name of the covariate.
        category: The category of the covariate.
        coords: The coordinates of the model.

    Returns:
        Either a tuple of integers corresponding to the index of the RV to select or
        `None` if the index could not be determined.

    Examples:
        >>> from vaxflux._util import _coord_index
        >>> coords = {
        ...     "season": ["2022/2023", "2023/2024"],
        ...     "covariate_names": ["sex", "age"],
        ...     "covariate_sex_categories": ["female", "male"],
        ...     "covariate_age_categories": ["youth", "adult", "senior"],
        ...     "covariate_age_categories_limited": ["adult", "senior"],
        ... }
        >>> _coord_index((), "2023/2024", None, None, coords)
        ()
        >>> _coord_index(("season",), "2023/2024", None, None, coords)
        (1,)
        >>> _coord_index(("covariate_names",), "2023/2024", "age", None, coords)
        (1,)
        >>> _coord_index(
        ...     ("covariate_age_categories",),
        ...     "2023/2024",
        ...     "age",
        ...     "senior",
        ...     coords,
        ... )
        (2,)
        >>> _coord_index(
        ...     ("covariate_age_categories_limited",),
        ...     "2023/2024",
        ...     "age",
        ...     "adult",
        ...     coords,
        ... )
        (0,)
        >>> _coord_index(
        ...     ("season", "covariate_names", "covariate_sex_categories"),
        ...     "2023/2024",
        ...     "sex",
        ...     "male",
        ...     coords,
        ... )
        (1, 0, 1)
        >>> _coord_index(
        ...     ("season", "covariate_names", "covariate_sex_categories"),
        ...     "2023/2024",
        ...     "sex",
        ...     "missing",
        ...     coords,
        ... ) is None
        True
    """
    try:
        return tuple(
            _coord_index_dim(dim, season, covariate_name, category, coords)
            for dim in dims
        )
    except ValueError:
        return None


def _compute_quantiles_and_max(
    data: pd.DataFrame,
    value_column: str,
    quantile_levels: Sequence[float] | npt.NDArray[np.float64],
    current_max: float,
) -> tuple[pd.DataFrame, float]:
    """
    Compute quantiles for a value column and update the max.

    Args:
        data: DataFrame containing a `mid_date` column and the target values.
        value_column: Column name to compute quantiles for.
        quantile_levels: Quantile levels to compute.
        current_max: Current maximum value to update.

    Returns:
        A tuple of the quantiles DataFrame and the updated maximum value.

    Examples:
        >>> import pandas as pd
        >>> from vaxflux._util import _compute_quantiles_and_max
        >>> data = pd.DataFrame(
        ...     {
        ...         "mid_date": pd.to_datetime(
        ...             [
        ...                 "2024-01-01",
        ...                 "2024-01-01",
        ...                 "2024-01-01",
        ...                 "2024-01-08",
        ...                 "2024-01-08",
        ...                 "2024-01-08",
        ...                 "2024-01-15",
        ...                 "2024-01-15",
        ...                 "2024-01-15",
        ...             ]
        ...         ),
        ...         "value": [8.0, 10.0, 16.0, 6.0, 12.0, 18.0, 9.0, 15.0, 21.0],
        ...     }
        ... )
        >>> quantiles, current_max = _compute_quantiles_and_max(
        ...     data, "value", [0.25, 0.5, 0.75], 14.0
        ... )
        >>> print(quantiles)  # doctest: +NORMALIZE_WHITESPACE
        level_1     0.25  0.50  0.75
        mid_date
        2024-01-01   9.0  10.0  13.0
        2024-01-08   9.0  12.0  15.0
        2024-01-15  12.0  15.0  18.0
        >>> current_max
        18.0
        >>> empty_quantiles, unchanged_max = _compute_quantiles_and_max(
        ...     data.iloc[0:0], "value", [0.5], 7.0
        ... )
        >>> print(empty_quantiles)
        Empty DataFrame
        Columns: []
        Index: []
        >>> unchanged_max
        7.0
        >>> no_quantiles, unchanged_max = _compute_quantiles_and_max(
        ...     data, "value", [], 20.0
        ... )
        >>> print(no_quantiles)
        Empty DataFrame
        Columns: []
        Index: []
        >>> unchanged_max
        20.0
    """
    if data.empty:
        return pd.DataFrame(), current_max
    quantile_values = np.array(quantile_levels, dtype=float)
    quantiles = (
        data.groupby("mid_date")[value_column]
        .quantile(quantile_values)
        .reset_index()
        .pivot_table(index="mid_date", columns="level_1", values=value_column)
    )
    if not quantiles.empty:
        current_max = max(current_max, float(quantiles.max().max()))
    return quantiles, current_max
