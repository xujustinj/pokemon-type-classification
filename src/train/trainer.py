from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

from model import Model, Config

M = TypeVar("M", bound=Model)


class Trainer(Generic[M], ABC):
    """
    A thin wrapper around a function that trains a model.
    """

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        config: Config
    ) -> M:
        """Train a model.

        Train a model given the X-values and y-values of the training data 
        and the model hyperparameters.

        Args:
            X_train (np.ndarray): X-values of training data
            y_train (np.ndarray): y-values of training data
            config (Config): model hyperparameters

        Returns:
            M: the trained model

        """
        pass
