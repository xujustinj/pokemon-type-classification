from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier

from model import RFModel
from .bayes import BayesSearchCV
from .tuner import Tuner, SearchSpace, Splitter


class RandomForestBayesTuner(Tuner[RFModel]):
    """
    Optimizes logistic regression through Bayesian optimization.
    """
    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        search: SearchSpace,
        split: Splitter,
    ) -> RFModel:
        opt = BayesSearchCV(
            estimator=SKRandomForestClassifier(
                random_state=441,
            ),
            scoring="accuracy",
            search_spaces=search,
            cv=split,
        )
        opt.fit(X_train, y_train)

        model: SKRandomForestClassifier = opt.best_estimator_  # type: ignore
        best_config: dict[str, Any] = opt.best_params_  # type: ignore
        return RFModel(model, config=best_config)
