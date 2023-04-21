from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression as SKLearnLogisticRegression
from sklearn.model_selection import BaseCrossValidator

from model import LogisticRegressionModel
from .bayes import BayesSearchCV
from .tuner import Tuner, SearchSpace, Splitter


class LogisticRegressionBayesOptTuner(Tuner[LogisticRegressionModel]):
    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        search: SearchSpace,
        split: BaseCrossValidator,
    ) -> LogisticRegressionModel:
        opt = BayesSearchCV(
            estimator=SKLearnLogisticRegression(),
            search_spaces=search,
            cv=split,
        )
        opt.fit(X_train, y_train)

        model: SKLearnLogisticRegression = opt.best_estimator_  # type: ignore
        best_config: dict[str, Any] = opt.best_params_  # type: ignore
        return LogisticRegressionModel(model, config=best_config)
