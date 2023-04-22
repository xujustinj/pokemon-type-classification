from sklearn.ensemble import RandomForestClassifier
import numpy as np

from model import Config, RFModel
from .trainer import Trainer


class RFTrainer(Trainer[RFModel]):
    def __init__(self):
        super().__init__()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        config: Config,
    ) -> RFModel:
        """Train a random forest classifier model.

        Train a random forest classifier model given the X-values and y-values of the training data 
        and the model hyperparameters.

        Args:
            X_train (np.ndarray): X-values of training data
            y_train (np.ndarray): y-values of training data
            config (Config): model hyperparameters

        Returns:
            RFModel: the trained random forest classifier model

        """
        n_estimators = config['n_estimators']
        criterion = config['criterion']
        max_features = config['max_features']
        model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                       max_features=max_features, random_state = 441)
        model.fit(X_train, y_train)

        return RFModel(model, config)