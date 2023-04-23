import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .model import Model, Config


class RandomForestModel(Model):
    """A random forest classification model.

    Implemented with sklearn.

    Args:
        model (RandomForest): The sklearn RandomForest model.
        config (Config): The hyperparameters of the model.
    """

    def __init__(self, model: RandomForestClassifier, config: Config):
        assert isinstance(model, RandomForestClassifier)

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

    def labels(self) -> np.ndarray:
        """The class names associated with the model.

        Returns:
            labels (ndarray): [K] Label names of the data, such that labels[k]
                is the name of the kth label.
        """
        return self._model.classes_
