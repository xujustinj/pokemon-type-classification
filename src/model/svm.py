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
        super().__init__(config)
        self._model = model

    def predict(self, X_test):
        return self._model.predict(X_test)

    def evaluate(self, X_test, y_test) -> float:
        y_pred = self._model.predict(X_test)
        return float(accuracy_score(y_test, y_pred))
    
    def labels(self):
        return self._model.classes_
