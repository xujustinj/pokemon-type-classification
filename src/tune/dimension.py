from typing import Any, Generic, Optional, TypeVar

import numpy as np
from skopt.space import Dimension, Categorical, Integer, Real

class Constant(Categorical):
    def __init__(self, value: Any):
        super().__init__(categories=[value])
        self.value = value


def get_value(dim: Dimension) -> Any:
    assert isinstance(dim, Categorical)
    assert len(dim.categories) == 1
    return dim.categories[0]


def discretize(dim: Dimension, steps: Optional[int] = None) -> list[Any]:
    if isinstance(dim, Categorical):
        return dim.categories

    assert isinstance(dim, Integer) or isinstance(dim, Real)

    if (
        isinstance(dim, Integer)
        and (steps is None or steps < dim.high + 1 - dim.low)
    ):
        return list(range(dim.low, dim.high + 1))
    assert steps is not None

    is_log = (dim.prior == "log-uniform")
    is_int = isinstance(dim, Integer)

    low = float(dim.low)
    high = float(dim.high)
    assert low < high
    if is_log:
        low = np.log(low)
        high = np.log(high)

    values = np.linspace(start=low, stop=high, num=steps)

    if is_log:
        values = np.exp(values)

    if is_int:
        values = np.round(values).astype(int)

    return list(values)
