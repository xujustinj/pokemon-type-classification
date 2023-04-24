from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from model import LogisticRegressionModel
from .bayes import BayesSearchCV
from .tuner import Tuner, SearchSpace, Splitter


class LogisticRegressionBayesTuner(Tuner[LogisticRegressionModel]):
    """
    Optimizes logistic regression through Bayesian optimization.
    """
    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        search: SearchSpace,
        split: Splitter,
    ) -> LogisticRegressionModel:
        """Tune hyperparameters of logistic regression models.

        Tune the model hyperparameters for logistic regression models 
        given the X-values and y-values of the training data. 

        Args:
            X_train (np.ndarray): X-values of training data
            y_train (np.ndarray): y-values of training data
            search (SearchSpace): The search space for hyperparameters
            split (Splitter): A cross validation generator

        Returns:
            LogisticRegressionModel: The trained logistic regression model
        """
        opt = BayesSearchCV(
            estimator=LogisticRegression(
                multi_class="multinomial",
                penalty="elasticnet",
                solver="saga",
                max_iter=9001,  # practically unlimited
                random_state=441,
                warm_start=True, # try to speed up optimization
            ),
            scoring="accuracy",
            search_spaces=search,
            cv=split,
        )
        opt.fit(X_train, y_train)

        model: LogisticRegression = opt.best_estimator_  # type: ignore
        best_config: dict[str, Any] = opt.best_params_  # type: ignore
        return LogisticRegressionModel(model, config=best_config)
