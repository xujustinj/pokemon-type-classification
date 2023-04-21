from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression as SKLogisticRegression

from .model import Model, Config


class LogisticRegressionModel(Model):
    def __init__(self, model: SKLogisticRegression, config: Config):
        assert isinstance(model, SKLogisticRegression)

        super().__init__(config)
        self._model = model

    def predict(self, X_test):
        return self._model.predict(X_test)

    def evaluate(self, X_test, y_test) -> float:
        y_pred = self._model.predict(X_test)
        return float(accuracy_score(y_test, y_pred))

    def labels(self):
        return self._model.classes_
