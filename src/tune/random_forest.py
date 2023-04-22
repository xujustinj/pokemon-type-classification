from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from model import RandomForestModel
from .bayes import BayesSearchCV
from .tuner import Tuner, SearchSpace, Splitter


class RandomForestBayesTuner(Tuner[RandomForestModel]):
    """
    Optimizes logistic regression through Bayesian optimization.
    """
    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        search: SearchSpace,
        split: Splitter,
    ) -> RandomForestModel:
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
