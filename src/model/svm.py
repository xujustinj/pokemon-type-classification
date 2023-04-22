# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 16:28:33 2023

@author: zhang
"""

from sklearn.metrics import accuracy_score
from sklearn import svm

from .model import Model, Config


class SVMModel(Model):
    def __init__(self, model: svm.SVC, config: Config):
        """Initialize a support vector classifier given model configuration and a support vector classifier instance.

        Initialize a support vector classifier model given model configuration and a support vector classifier instance.

        Args:
            model (ensemble.RandomForestClassifier): A support vector classifier model instance
            config (Config): A config specifying the hyperparameters for each model

        """
        super().__init__(config)
        self._model = model

    def predict(self, X_test):
        """Predict y-values given X-values.

        Predict y-values given X-values using the model.

        Args:
            X_test (numpy.ndarray): X-values of test data

        Returns:
            numpy.ndarray: Predicted y-values

        """
        return self._model.predict(X_test)

    def evaluate(self, X_test, y_test) -> float:
        """Evaluate the model with test data.

        Evaluate the model given X-values and y-values of test data.

        Args:
            X-values (numpy.ndarray): X-values of test data
            y-values (numpy.ndarray): y-values of test data

        Returns:
            float: Accuracy of the model prediction on test data

        """
        y_pred = self._model.predict(X_test)
        return float(accuracy_score(y_test, y_pred))
    
    def labels(self):
        """Return the distinct classes of data used to train the model.

        Return the distinct label classes of data used to train the model.

        Returns:
            numpy.ndarray: the distinct label classes of data used to train the model

        """
        return self._model.classes_
