from sklearn.metrics import accuracy_score
from sklearn import ensemble

from .model import Model, Config


class RFModel(Model):
    def __init__(self, model: ensemble.RandomForestClassifier, config: Config):
        super().__init__(config)
        self._model = model

    def predict(self, X_test):
        return self._model.predict(X_test)

    def evaluate(self, X_test, y_test) -> float:
        y_pred = self._model.predict(X_test)
        return float(accuracy_score(y_test, y_pred))
    
    def labels(self):
        return self._model.classes_
