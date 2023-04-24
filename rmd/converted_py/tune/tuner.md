## `tune/tuner.py`

```{py}
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeAlias, TypeVar

import numpy as np
from sklearn.model_selection import BaseCrossValidator
from skopt.space import Dimension

from model import Model

SearchSpaceDict: TypeAlias = dict[str, Dimension]
SearchSpaceList: TypeAlias = list[SearchSpaceDict]
SearchSpaceWeightedList: TypeAlias = list[tuple[SearchSpaceDict, int]]
SearchSpace: TypeAlias = SearchSpaceDict | SearchSpaceList | SearchSpaceWeightedList

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
        """Tune hyperparameters of a model using cross-validation.

        Args:
            X_train (ndarray): X-values of the training data.
            y_train (ndarray): y-values of the training data.
            search (SearchSpace): The search space for the hyperparameters.
            split (Splitter): A cross validation generator.

        Returns:
            model: A fitted model with the optimal hyperparameters.
        """
        pass
```
