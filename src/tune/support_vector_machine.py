from typing import Any

import numpy as np
from sklearn.svm import SVC as SKSupportVectorClassifier

from model import SVMModel
from .bayes import BayesSearchCV
from .tuner import Tuner, SearchSpace, Splitter


class SupportVectorMachineBayesTuner(Tuner[SVMModel]):
    """
    Optimizes logistic regression through Bayesian optimization.
    """
    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        search: SearchSpace,
        split: Splitter,
    ) -> SVMModel:
        opt = BayesSearchCV(
            estimator=SKSupportVectorClassifier(
                probability=True,
                random_state=441,
                decision_function_shape="ovr",
            ),
            scoring="accuracy",
            search_spaces=search,
            cv=split,
        )
        opt.fit(X_train, y_train)

        model: SKSupportVectorClassifier = opt.best_estimator_  # type: ignore
        best_config: dict[str, Any] = opt.best_params_  # type: ignore
        return SVMModel(model, config=best_config)
