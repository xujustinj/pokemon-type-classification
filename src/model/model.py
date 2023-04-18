# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 22:54:57 2023

@author: zhang
"""

from abc import ABC, abstractmethod
import joblib
from typing import Any, Type, TypeVar

import numpy as np


Self = TypeVar("Self", bound="Model")
Config = dict[str, Any]


class Model(ABC):
    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        pass

    @abstractmethod
    def labels(self) -> np.ndarray:
        pass

    def save(self, path: str) -> None:
        """
        Saves the model to the given path, such that calling load on that same
        path returns an identical model.
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls: Type[Self], path: str) -> Self:
        model = joblib.load(path)
        assert isinstance(model, cls)
        return model
