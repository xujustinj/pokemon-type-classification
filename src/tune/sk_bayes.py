import os
from typing import Any, Type

import numpy as np
from sklearn.base import BaseEstimator
from skopt import BayesSearchCV

from model import SKModel, Config, wrap_sklearn_classifier_via_duplication
from .tuner import Tuner, SearchSpace, Splitter


class SKBayesTuner(Tuner[SKModel]):
    """Optimizes an sklearn classifier through Bayesian optimization.

    Args:
        Classifier: An sklearn classifier model class (e.g. LogisticRegression).
            Must support the fit, predict, and predict_proba methods.
        duplicate (bool): Whether to wrap the classifier as a multiset
            classifier via the duplication method.
        num_labels (int): If duplicate=True, the number of labels that the
            multiset classifier should predict.
    """
    def __init__(
        self,
        Classifier: Type[BaseEstimator],
        duplicate: bool = False,
        num_labels: int = 2,
    ):
        if duplicate:
            Classifier = wrap_sklearn_classifier_via_duplication(
                Classifier=Classifier,
                num_labels=num_labels
            )
        self._estimator = Classifier()

        # use one fewer core to avoid hogging all resources
        self._cores = max(len(os.sched_getaffinity(0)) - 1, 1)
        print(f"Using {self._cores} core(s)")

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
            n_jobs=self._cores,
        )
        opt.fit(X_train, y_train)

        model: BaseEstimator = opt.best_estimator_  # type: ignore
        best_config: dict[str, Any] = opt.best_params_  # type: ignore
        return SKModel(model, config=best_config)
