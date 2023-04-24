## `tune/sk_bayes.py`

```{py}
from typing import Any, Generic

import numpy as np
from sklearn.base import BaseEstimator
from skopt import BayesSearchCV

from model import SKClassifier, SKModel
from util import PARALLELISM
from .tuner import Tuner, SearchSpace, Splitter


class SKBayesTuner(Tuner[SKModel], Generic[SKClassifier]):
    """Optimizes an sklearn classifier through Bayesian optimization.

    Args:
        Classifier: An sklearn classifier model class (e.g. LogisticRegression).
            Must support the fit, predict, and predict_proba methods.
        duplicate (bool): Whether to wrap the classifier as a multiset
            classifier via the duplication method.
        num_labels (int): If duplicate=True, the number of labels that the
            multiset classifier should predict.
    """
    def __init__(self, estimator: SKClassifier):
        self._estimator = estimator

    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        search: SearchSpace,
        split: Splitter,
    ) -> SKModel:
        """Tune hyperparameters of a model using cross-validation.

        Performs Bayesian optimization to efficiently sample the search space.

        Args:
            X_train (ndarray): X-values of the training data.
            y_train (ndarray): y-values of the training data.
            search (SearchSpace): The search space for the hyperparameters.
            split (Splitter): A cross validation generator.

        Returns:
            model: A fitted model with the optimal hyperparameters.
        """
        opt = BayesSearchCV(
            estimator=self._estimator,
            search_spaces=search,
            cv=split,
            n_jobs=PARALLELISM,
        )
        opt.fit(X_train, y_train)

        model: BaseEstimator = opt.best_estimator_  # type: ignore
        best_config: dict[str, Any] = opt.best_params_  # type: ignore
        return SKModel(model, config=best_config)
```
