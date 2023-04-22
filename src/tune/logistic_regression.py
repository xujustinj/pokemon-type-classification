from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression as SKLearnLogisticRegression, LogisticRegressionCV

from model import LogisticRegressionModel
from .bayes import BayesSearchCV
from .dimension import Real, discretize, get_value
from .tuner import Tuner, SearchSpace, Splitter


_LOGISTIC_REGRESSION_OPTIONS = dict(
    multi_class="multinomial",
    penalty="elasticnet",
    solver="saga",
    max_iter=9001,  # practically unlimited
    random_state=441,
)


class LogisticRegressionBayesTuner(Tuner[LogisticRegressionModel]):
    """
    Optimizes logistic regression through Bayesian Optimization.
    """
    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        search: SearchSpace,
        split: Splitter,
    ) -> LogisticRegressionModel:
        opt = BayesSearchCV(
            estimator=SKLearnLogisticRegression(
                **_LOGISTIC_REGRESSION_OPTIONS,
                warm_start=True, # try to speed up optimization
            ),
            scoring="accuracy",
            search_spaces=search,
            cv=split,
        )
        opt.fit(X_train, y_train)

        model: SKLearnLogisticRegression = opt.best_estimator_  # type: ignore
        best_config: dict[str, Any] = opt.best_params_  # type: ignore
        return LogisticRegressionModel(model, config=best_config)


class LogisticRegressionGridTuner(Tuner[LogisticRegressionModel]):
    """
    Optimizes logistic regression through grid search.
    """
    def __init__(
        self,
        steps_C: int,
        steps_l1_ratio: int,
    ):
        self._steps_C = steps_C
        self._steps_l1_ratio = steps_l1_ratio

    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        search: SearchSpace,
        split: Splitter,
    ) -> LogisticRegressionModel:
        assert isinstance(search, dict)

        C = search.pop("C", Real(low=1e-4, high=1e4, prior="log-uniform"))
        l1_ratio = search.pop("l1_ratio", Real(low=0.0, high=1.0, prior="uniform"))

        model = LogisticRegressionCV(
            scoring="accuracy",
            cv=split,
            Cs=discretize(C, steps=self._steps_C),
            l1_ratios=discretize(l1_ratio, steps=self._steps_l1_ratio),
            **_LOGISTIC_REGRESSION_OPTIONS,
            **{key: get_value(dim) for key, dim in search.items()}
        )
        model.fit(X_train, y_train)

        config = dict(C=model.C_, l1_ratio=model.l1_ratio_)
        return LogisticRegressionModel(model, config=config)
