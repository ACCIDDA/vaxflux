__all__: tuple[str, ...] = ()


import re
from typing import Annotated, Any, Callable, overload

from pydantic import BeforeValidator


_CLEAN_TEXT_REGEX = re.compile(r"[^a-zA-Z0-9]")


def _clean_text(text: str) -> str:
    """
    Clean text by removing non-alphanumeric characters and converting to title case.

    Args:
        text: The text to clean.

    Returns:
        The cleaned text.

    Examples:
        >>> from vaxflux._util import _clean_text
        >>> _clean_text("Abc Def$$GHI%jkl")
        'AbcDefGhiJkl'
    """
    return re.sub(r"[^a-zA-Z0-9]", " ", text).title().replace(" ", "")


def _clean_name(
    *args: str | None, joiner: str = "", transform: Callable[[str], str] = lambda x: x
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
        >>> _clean_name("Abc", "Def", "GHI", joiner="_", transform=lambda x: x.lower())
        'abc_def_ghi'
        >>> _clean_name("Abc", "Def", "GHI", joiner="_", transform=lambda x: x.upper())
        'ABC_DEF_GHI'
        >>> _clean_name("Abc", "Def", "GHI", joiner="_", transform=lambda x: x.title())
        'Abc_Def_Ghi'
        >>> _clean_name("a$b", "c#d", "e99", joiner="_")
        'a_b_c_d_e99'
    """
    return joiner.join(
        map(
            lambda x: transform(_CLEAN_TEXT_REGEX.sub(" ", x).strip()).replace(
                " ", joiner
            ),
            filter(None, args),
        )
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
def _make_float_list(x: Any) -> Any: ...


def _make_float_list(x: float | int | Any) -> list[float] | Any:
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
    return [float(x)] if isinstance(x, (int, float)) else x


ListOfFloats = Annotated[list[float], BeforeValidator(_make_float_list)]
