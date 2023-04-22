# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 16:33:29 2023

@author: zhang
"""

from sklearn import svm
import numpy as np

from model import Config, SVMModel
from .trainer import Trainer


class SVMTrainer(Trainer[SVMModel]):
    def __init__(self):
        super().__init__()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        config: Config,
    ) -> SVMModel:
        """Train a support vector classifier model.

        Train a support vector classifier model given the X-values and y-values of the training data 
        and the model hyperparameters.

        Args:
            X_train (np.ndarray): X-values of training data
            y_train (np.ndarray): y-values of training data
            config (Config): model hyperparameters

        Returns:
            SVMModel: the trained support vector classifier model

        """
        C = config['C']
        kernel = config['kernel']
        degree = config['degree']
        gamma = config['gamma']
        decision_function_shape = config['decision_function_shape']
        probability = config['probability']
        model = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma,
                        probability=probability,
                        decision_function_shape=decision_function_shape)

        model.fit(X_train, y_train)

        return SVMModel(model, config)
