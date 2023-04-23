import numpy as np
from sklearn.linear_model import LogisticRegression

from .model import Model, Config


class LogisticRegressionModel(Model):
    """A logistic regression classification model.

    Implemented with sklearn, including ElasticNet, L1, and L2 regularization.

    Args:
        model (LogisticRegression): The sklearn LogisticRegression model.
        config (Config): The hyperparameters of the model.
    """

    def __init__(self, model: LogisticRegression, config: Config):
        assert isinstance(model, LogisticRegression)

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
