from typing import Generic, TypeVar

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier as SKRandomForest
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.svm import SVC as SKSupportVectorMachine

from .model import Model, Config

SKClassifier = TypeVar("SKClassifier", bound=BaseEstimator)

class SKModel(Model, Generic[SKClassifier]):
    """Wrapper for an sklearn model.

    Args:
        model (SKClassifier): An sklearn classification model. Must support
            predict() and predict_proba() methods. e.g.: LogisticRegression
        config (Config): The hyperparameters of the model.
    """

    def __init__(self, model: SKClassifier, config: Config):
        assert isinstance(model, BaseEstimator)

        super().__init__(config)
        self._model = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X (ndarray): [N x P] Predictor values.

        Returns:
            y (ndarray): [N] Predicted class labels.
        """
        return self._model.predict(X)

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X (ndarray): [N x P] Predictor values.

        Returns:
            p (ndarray): [N x K] Predicted probabilities, where p[i, k] is the
                probability that X[i] is class k.
        """
        return self._model.predict_proba(X)

    @property
    def labels(self) -> np.ndarray:
        """The class names associated with the model.

        Returns:
            labels (ndarray): [K] Label names of the data, such that labels[k]
                is the name of the kth label.
        """
        return self._model.classes_
