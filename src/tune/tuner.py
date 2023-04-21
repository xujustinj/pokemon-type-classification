from abc import ABC, abstractmethod
from typing import Any, Generic, TypeAlias, TypeVar

import numpy as np
from sklearn.model_selection import BaseCrossValidator

from model import Model

Dimension: TypeAlias = tuple | list[Any]

SKSearchSpaceDict: TypeAlias = dict[str, Dimension]
SKSearchSpaceList: TypeAlias = list[SKSearchSpaceDict |
                                    tuple[SKSearchSpaceDict, int]]
SearchSpace: TypeAlias = SKSearchSpaceDict | SKSearchSpaceList

Splitter: TypeAlias = BaseCrossValidator

M = TypeVar("M", bound=Model)


class Tuner(Generic[M], ABC):
    """
    A thin wrapper around a function that selects the best model given some
    cross validation splits.
    """

    @abstractmethod
    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        search: SearchSpace,
        split: Splitter,
    ) -> M:
        pass
