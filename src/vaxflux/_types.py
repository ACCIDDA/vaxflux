from typing import Any

import numpy as np
from jax import Array as JaxArray

NumericalArrayLike = (
    JaxArray | np.ndarray[Any, np.dtype[np.number[Any]]] | int | float | complex
)
