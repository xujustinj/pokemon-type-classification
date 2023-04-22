from typing import Any

from skopt.space import Dimension, Categorical, Integer, Real

class Constant(Categorical):
    def __init__(self, value: Any):
        super().__init__(categories=[value])
        self.value = value
