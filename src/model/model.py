from abc import ABC, abstractmethod
import joblib
from typing import Any, Type, TypeVar

from sklearn.metrics import accuracy_score

import numpy as np


Self = TypeVar("Self", bound="Model")
Config = dict[str, Any]


class Model(ABC):
    """An abstract wrapper for a K-class single-label classification model.

    Predicts on P-dimensional inputs. That is, R^P -> {1, ..., K}.

    Args:
        config (Config): The hyperparameters of the model.
    """

    def __init__(self, config: Config):
        assert isinstance(config, dict)
        for k in config.keys():
            assert isinstance(k, str)

        self.config = config

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X (ndarray): [N x P] Predictor values.

        Returns:
            y (ndarray): [N] Predicted class labels.
        """
        pass

    @abstractmethod
    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X (ndarray): [N x P] Predictor values.

        Returns:
            p (ndarray): [N x K] Predicted probabilities, where p[i, k] is the
                probability that X[i] is class k.
        """
        pass

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate the accuracy of the model with test data.

        Args:
            X (ndarray): [N x P] Predictor values.
            y (ndarray): [N] True class labels.

        Returns:
            accuracy (float): Accuracy of the model prediction on test data.
        """
        return float(accuracy_score(y_true=y, y_pred=self.predict(X)))

    @abstractmethod
    def labels(self) -> np.ndarray:
        """The class names associated with the model.

        Returns:
            labels (ndarray): [K] Label names of the data, such that labels[k]
                is the name of the kth label.
        """
        pass

    def save(self, path: str) -> None:
        """Save the model to the specified path.

        Args:
            path (str): The path where the model will be stored.
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls: Type[Self], path: str) -> Self:
        """Load model from the specified path.

        Args:
            path (str): The path to the model file.

        Returns:
            model (Self): A model of this class's type.
        """
        model = joblib.load(path)
        assert isinstance(model, cls)
        return model
