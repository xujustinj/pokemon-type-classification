from abc import ABC, abstractmethod
import joblib
from typing import Any, Type, TypeVar

import numpy as np


Self = TypeVar("Self", bound="Model")
Config = dict[str, Any]


class Model(ABC):
    """An abstract K-class M-label multiset classification model.

    Predicts on P-dimensional inputs. That is, R^P -> {1, ..., K}^M.
    When M=1, this is regular single-label classification.

    Args:
        config (Config): The hyperparameters of the model.
    """

    def __init__(self, config: Config):
        assert isinstance(config, dict)
        for k in config.keys():
            assert isinstance(k, str)

        self.config = config

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X (ndarray): [N x P] predictor values.

        Returns:
            y (ndarray): [N] (if M=1) or [N x M] predicted class labels.
        """
        p = self.predict_probabilities(X)
        return self.labels[p.argmax(axis=-1)]

    @abstractmethod
    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        When M>=2, it is more accurate to think of the probabilities as the
        "expected multiplicity" of each class, so it is possible for a value
        larger than 1 to reflect an element that occurs multiple times. This is
        consistent with single-label classification when M=1.

        The total probabilities must sum to M.

        Args:
            X (ndarray): [N x P] predictor values.

        Returns:
            p (ndarray): [N x K] predicted "probabilities", where p[i, k] is the
                expected multiplicity of class k in y[i].
        """
        pass

    @property
    @abstractmethod
    def labels(self) -> np.ndarray:
        """The label names associated with the model.

        Returns:
            labels (ndarray): [K] label names of the data, such that labels[k]
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
            model (Model): A model of this class's type.
        """
        model = joblib.load(path)
        assert isinstance(model, cls)
        return model
