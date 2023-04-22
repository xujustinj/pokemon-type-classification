from abc import ABC, abstractmethod
import joblib
from typing import Any, Type, TypeVar

import numpy as np


Self = TypeVar("Self", bound="Model")
Config = dict[str, Any]


class Model(ABC):
    def __init__(self, config: Config):
        """Initialize a model instance given model configuration.

        Initialize a model instance given model configuration.

        Args:
            config (Config): A config specifying the hyperparameters for each model

        """
        assert isinstance(config, dict)
        for k in config.keys():
            assert isinstance(k, str)

        self.config = config

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict y-values given X-values.

        Predict y-values given X-values using the model.

        Args:
            X_test (numpy.ndarray): X-values of test data

        Returns:
            numpy.ndarray: Predicted y-values

        """
        pass

    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Evaluate the model with test data.

        Evaluate the model given X-values and y-values of test data.

        Args:
            X-values (numpy.ndarray): X-values of test data
            y-values (numpy.ndarray): y-values of test data

        Returns:
            float: Accuracy of the model prediction on test data

        """
        pass

    @abstractmethod
    def labels(self) -> np.ndarray:
        """Return the distinct classes of data used to train the model.

        Return the distinct label classes of data used to train the model.

        Returns:
            numpy.ndarray: the distinct label classes of data used to train the model

        """
        pass

    def save(self, path: str) -> None:
        """Save the model to the specified path.

        Save the model to the specified path. The directories must exist already.

        Args:
            path (str): The path, including the filename, where the model will be stored

        """
        joblib.dump(self, path)

    @classmethod
    def load(cls: Type[Self], path: str) -> Self:
        """Load model given the model path.

        Load the model given the model path with joblib.

        Args:
            path (str): The path, including the filename, where the model will be stored

        """
        model = joblib.load(path)
        assert isinstance(model, cls)
        return model
