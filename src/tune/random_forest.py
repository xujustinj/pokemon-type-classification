from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from model import RandomForestModel
from .bayes import BayesSearchCV
from .tuner import Tuner, SearchSpace, Splitter


class RandomForestBayesTuner(Tuner[RandomForestModel]):
    """
    Optimizes random forest through Bayesian optimization.
    """
    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        search: SearchSpace,
        split: Splitter,
    ) -> RandomForestModel:
        """Tune hyperparameters of random forest models.

        Tune the model hyperparameters for random forest models 
        given the X-values and y-values of the training data. 

        Args:
            X_train (np.ndarray): X-values of training data
            y_train (np.ndarray): y-values of training data
            search (SearchSpace): The search space for hyperparameters
            split (Splitter): A cross validation generator

        Returns:
            RandomForestModel: The trained random forest model
        """
        opt = BayesSearchCV(
            estimator=RandomForestClassifier(
                random_state=441,
            ),
            scoring="accuracy",
            search_spaces=search,
            cv=split,
        )
        opt.fit(X_train, y_train)

        model: RandomForestClassifier = opt.best_estimator_  # type: ignore
        best_config: dict[str, Any] = opt.best_params_  # type: ignore
        return RandomForestModel(model, config=best_config)
