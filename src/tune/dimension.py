from typing import Any

from skopt.space import Dimension, Categorical, Integer, Real

class Constant(Categorical):
    """A search space dimension that has only one value.

    Args:
        value: The single value of the search space dimension.
    """
    def __init__(self, value: Any):
        super().__init__(categories=[value])
        self.value = value
