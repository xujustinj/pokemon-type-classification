from typing import Type

import numpy as np
from sklearn.base import BaseEstimator

from util import multiset_accuracy


def wrap_sklearn_classifier_via_duplication(
    Classifier: Type[BaseEstimator],
    num_labels: int = 2,
) -> Type[BaseEstimator]:
    """Converts a single-label sklearn classifier to a multiset classifier.

    Implements the duplication method by intercepting calls to fit and predict
    on the wrapped classifier. When fitting, it converts a set of multi-label
    training observations into num_labels sets of single-label training
    observations. When predicting, it selects the top num_labels labels (as a
    multiset) instead of predicting a single label.

    Args:
        Classifier: An sklearn classifier model class (e.g. LogisticRegression).
            Must support the fit, predict, and predict_proba methods.
        num_labels (int): The number of labels that the multiset classifier
            should predict.

    Returns:
        DuplicationClassifier: A classifier model class that performs multiset
            classification with num_labels labels. It is fully compatible with
            the sklearn estimator API, so it can be passed into any function
            that accepts an sklearn model.
    """
    class DuplicationClassifier(Classifier):
        def fit(self, X: np.ndarray, y: np.ndarray) -> None:
            """Fit the model on the given data.

            Each observation is duplicated num_labels times, such that

            """
            dup_X = X.repeat(repeats=num_labels, axis=0)
            dup_y = y.flatten()
            super().fit(dup_X, dup_y)

        def predict(self, X: np.ndarray) -> np.ndarray:
            """Predict num_labels for each observation.
            """
            p = self.predict_proba(X)
            all_label_ids = []
            for _ in range(num_labels):
                label_ids = p.argmax(axis=-1)
                for i, k in enumerate(label_ids):
                    p[i,k] -= 1
                all_label_ids.append(label_ids)
            return self.classes_[np.stack(all_label_ids, axis=-1)]

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            """Predict the expected
            """
            return num_labels * super().predict_proba(X)

        def score(self, X: np.ndarray, y: np.ndarray) -> float:
            return multiset_accuracy(y_true=y, y_pred=self.predict(X))

    return DuplicationClassifier
