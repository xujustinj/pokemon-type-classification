from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression as SKLogisticRegression

from model import Config, LogisticRegressionModel
from .trainer import Trainer


MultiClass = Literal["auto", "ovr", "multinomial"]
Penalty = Literal["l1", "l2", "elasticnet"] | None


class LRTrainer(Trainer[LogisticRegressionModel]):
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        config: Config,
    ) -> LogisticRegressionModel:
        """Train a logistic regression model.

        Train a logistic regression model given the X-values and y-values of the training data 
        and the model hyperparameters.

        Args:
            X_train (np.ndarray): X-values of training data
            y_train (np.ndarray): y-values of training data
            config (Config): model hyperparameters

        Returns:
            LogisticRegressionModel: the trained logistic regression model

        """
        multi_class: MultiClass = config.get("multi_class", "auto")
        penalty: Penalty = config.get("penalty")
        C: float = config.get("C", 1.0)

        model = SKLogisticRegression(
            multi_class=multi_class,
            penalty=penalty,
            C=C,
            max_iter=9001,  # practically unlimited iterations
            random_state=441,
        )

        model.fit(X_train, y_train)

        return LogisticRegressionModel(model=model, config=config)
