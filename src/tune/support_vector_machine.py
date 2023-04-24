from typing import Any

import numpy as np
from sklearn.svm import SVC

from model import SupportVectorMachineModel
from .bayes import BayesSearchCV
from .tuner import Tuner, SearchSpace, Splitter


class SupportVectorMachineBayesTuner(Tuner[SupportVectorMachineModel]):
    """
    Optimizes support vector machine through Bayesian optimization.
    """
    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        search: SearchSpace,
        split: Splitter,
    ) -> SupportVectorMachineModel:
        """Tune hyperparameters of support vector machine models.

        Tune the model hyperparameters for support vector machine models 
        given the X-values and y-values of the training data. 

        Args:
            X_train (np.ndarray): X-values of training data
            y_train (np.ndarray): y-values of training data
            search (SearchSpace): The search space for hyperparameters
            split (Splitter): A cross validation generator

        Returns:
            SupportVectorMachineModel: The trained support vector machine model
        """
        opt = BayesSearchCV(
            estimator=SVC(
                probability=True,
                random_state=441,
                decision_function_shape="ovr",
            ),
            scoring="accuracy",
            search_spaces=search,
            cv=split,
        )
        opt.fit(X_train, y_train)

        model: SVC = opt.best_estimator_  # type: ignore
        best_config: dict[str, Any] = opt.best_params_  # type: ignore
        return SupportVectorMachineModel(model, config=best_config)
