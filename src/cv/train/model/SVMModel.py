# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 16:28:33 2023

@author: zhang
"""

from .modelBase import Model
import joblib
from sklearn.metrics import accuracy_score

class SVMModel(Model): 
    
    def __init__(self, model):
        self.model = model
    
    def predict(self, X_test):
        self.model.predict(X_test)
    
    def save(self, path):
        joblib.dump(self.model, path)
    
    def evaluate(self, X_test, y_test): 
        y_pred = self.model.predict(X_test)
        return accuracy_score(y_test, y_pred)
    
    def load(self, path): 
        self.model = joblib.load(path)
        