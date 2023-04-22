from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from .model import Model, Config


class LogisticRegressionModel(Model):
    def __init__(self, model: LogisticRegression, config: Config):
        """Initialize a logistic regression model given model configuration and a logistic regression model instance.

        Initialize a logistic regression model given model configuration and a logistic regression model instance.

        Args:
            model (LogisticRegression): A logistic regression model instance
            config (Config): A config specifying the hyperparameters for each model

        """
        assert isinstance(model, LogisticRegression)

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
