from typing import Any, cast

import arviz as az


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
