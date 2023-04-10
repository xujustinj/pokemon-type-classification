# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 16:33:29 2023

@author: zhang
"""

from sklearn import svm
import numpy as np
import pandas as pd
from .model.SVMModel import SVMModel

def train_SVM_model(X_train, y_train, model_config): 
    C = model_config['C']
    kernel = model_config['kernel']
    degree = model_config['degree']
    gamma = model_config['gamma']
    decision_function_shape = model_config['decision_function_shape']
    probability = model_config['probability']
    model = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, 
                    probability=probability,
                    decision_function_shape=decision_function_shape)
    
    
    model.fit(X_train, y_train)
    
    return SVMModel(model)
