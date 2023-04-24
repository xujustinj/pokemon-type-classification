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
        """Tune hyperparameters of models of type M.

        Tune the model hyperparameters for models of type M 
        given the X-values and y-values of the training data. 

        Args:
            X_train (np.ndarray): X-values of training data
            y_train (np.ndarray): y-values of training data
            search (SearchSpace): The search space for hyperparameters
            split (Splitter): A cross validation generator

        Returns:
            M: The trained model of type M
        """
        pass
