"""Covariate category utilities and covariate model exports."""

__all__ = []

from itertools import product
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CovariateCategories(BaseModel):
    """A representation of the categories for a covariate."""

    model_config = ConfigDict(frozen=True)

    #: The name of the covariate.
    covariate: str

    #: The categories for the covariate.
    categories: Annotated[tuple[str, ...], Field(min_length=2)]

    @field_validator("categories", mode="after")
    @classmethod
    def _are_categories_unique(cls, categories: tuple[str, ...]) -> tuple[str, ...]:
        unique_categories: set[str] = set()
        for category in categories:
            if category in unique_categories:
                msg = f"Category '{category}' is not unique."
                raise ValueError(msg)
            unique_categories.add(category)
        return categories


def _covariate_categories_to_dict(
    covariate_categories: list[CovariateCategories],
) -> dict[str, list[str]]:
    """
    Convert the covariate categories to a dictionary.

    Args:
        covariate_categories: The covariate categories to convert.

    Returns:
        A dictionary with the covariate categories, where the keys are the covariate
        names and the values are the categories.

    Raises:
        ValueError: If the covariate names are not unique.
    """
    covariate_categories_dict = {
        covariate_category.covariate: list(covariate_category.categories)
        for covariate_category in covariate_categories
    }
    if len(covariate_categories_dict) != len(covariate_categories):
        msg = (
            "Covariate category `covariate` names must be unique to collapse to a dict."
        )
        raise ValueError(msg)
    return covariate_categories_dict


def _covariate_categories_product(
    covariate_categories: list[CovariateCategories],
) -> list[dict[str, str]]:
    """
    Create a product of the covariate categories.

    Args:
        covariate_categories: The covariate categories to use.

    Returns:
        A list of dictionaries with the product of the covariate categories.
    """
    if not covariate_categories:
        return []
    names = [
        covariate_category.covariate for covariate_category in covariate_categories
    ]
    categories = [
        covariate_category.categories for covariate_category in covariate_categories
    ]
    return [
        dict(zip(names, category, strict=False))
        for category in list(product(*categories))
    ]
