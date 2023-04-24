import numpy as np

from util import duplicate, multiset_accuracy, predict_multiset_indices
from .sk_model import (
    SKLogisticRegression,
    SKSupportVectorMachine,
    SKRandomForest,
)


class DuplicateSKRandomForest(SKRandomForest):
    """A multiset random forest classifier using the duplication method.

    Implements the duplication method by intercepting calls to fit and predict
    on the wrapped classifier. When fitting, it converts a set of multi-label
    training observations into num_labels sets of single-label training
    observations. When predicting, it selects the top num_labels labels (as a
    multiset) instead of predicting a single label.
    """

    @property
    def cardinality(self) -> int:
        """The cardinality of the multisets predicted by the model.

        Override this for different cardinality.

        Returns:
            cardinality (int): The size (counting duplicates) of the multiset of
                labels predicted by the model.
        """
        return 2

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model on the given data.

        Each observation is duplicated self.cardinality times, such that

        """
        X, y = duplicate(X, y)
        super().fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict num_labels for each observation.
        """
        p = self.predict_proba(X)
        ids = predict_multiset_indices(p, cardinality=self.cardinality)
        return self.classes_[ids]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict the expected multiplicity of each label.
        """
        return self.cardinality * super().predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return multiset_accuracy(y_true=y, y_pred=self.predict(X))


class DuplicateSKLogisticRegression(SKLogisticRegression):
    """A multiset logistic regression classifier using the duplication method.

    Implements the duplication method by intercepting calls to fit and predict
    on the wrapped classifier. When fitting, it converts a set of multi-label
    training observations into num_labels sets of single-label training
    observations. When predicting, it selects the top num_labels labels (as a
    multiset) instead of predicting a single label.
    """

    @property
    def cardinality(self) -> int:
        """The cardinality of the multisets predicted by the model.

        Override this for different cardinality.

        Returns:
            cardinality (int): The size (counting duplicates) of the multiset of
                labels predicted by the model.
        """
        return 2

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model on the given data.

        Each observation is duplicated self.cardinality times, such that

        """
        X, y = duplicate(X, y)
        super().fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict num_labels for each observation.
        """
        p = self.predict_proba(X)
        ids = predict_multiset_indices(p, cardinality=self.cardinality)
        return self.classes_[ids]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict the expected multiplicity of each label.
        """
        return self.cardinality * super().predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return multiset_accuracy(y_true=y, y_pred=self.predict(X))


class DuplicateSKSupportVectorMachine(SKSupportVectorMachine):
    """A multiset support vector classifier using the duplication method.

    Implements the duplication method by intercepting calls to fit and predict
    on the wrapped classifier. When fitting, it converts a set of multi-label
    training observations into num_labels sets of single-label training
    observations. When predicting, it selects the top num_labels labels (as a
    multiset) instead of predicting a single label.
    """

    @property
    def cardinality(self) -> int:
        """The cardinality of the multisets predicted by the model.

        Override this for different cardinality.

        Returns:
            cardinality (int): The size (counting duplicates) of the multiset of
                labels predicted by the model.
        """
        return 2

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model on the given data.

        Each observation is duplicated self.cardinality times, such that

        """
        X, y = duplicate(X, y)
        super().fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict num_labels for each observation.
        """
        p = self.predict_proba(X)
        ids = predict_multiset_indices(p, cardinality=self.cardinality)
        return self.classes_[ids]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict the expected multiplicity of each label.
        """
        return self.cardinality * super().predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return multiset_accuracy(y_true=y, y_pred=self.predict(X))
