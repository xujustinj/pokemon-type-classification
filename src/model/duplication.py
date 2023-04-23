import numpy as np

from .model import Model

class DuplicationModel(Model):
    """Duplicates a K-class single-label classification model M > 1 times.

    Args:
        model (Model): The inner single-label classifier.
    """

    def __init__(self, model: Model, M: int = 2):
        assert isinstance(model, Model)
        assert M > 1

        super().__init__(config=model.config)
        self._model = model
        self._M = M

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        For example, if [2.2, 0.5, 1.3, 0.0, 0.0] are the probabilities of a
        5-class 4-label classifier, then the predicted classes will be
            0   since p[0]=2.2 is the highest
                we subtract 1 from p[0] to get 1.2
            2   since p[2]=1.3 is the highest
                we subtract 1 from p[2] to get 0.3
            0   since p[0]=1.2 is the highest
                we subtract 1 from p[0] to get 0.2
            1   since p[1]=0.5 is the highest
                we subtract 1 from p[1] to get -0.5

        Args:
            X (ndarray): [N x P] Predictor values.

        Returns:
            y (ndarray): [N x M] Predicted class labels. Classes in reported in
                order from highest to lowest probability.
        """
        p = self.predict_probabilities(X)
        all_label_ids = []
        for _ in range(self._M):
            label_ids = p.argmax(axis=-1)
            for i, k in enumerate(label_ids):
                p[i,k] -= 1
            all_label_ids.append(label_ids)
        return self.labels[np.stack(all_label_ids, axis=-1)]

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        When M>=2, it is more accurate to think of the probabilities as the
        "expected multiplicity" of each class, so it is possible for a value
        larger than 1 to reflect an element that occurs multiple times. This is
        consistent with single-label classification when M=1.

        The total probabilities must sum to M.

        Args:
            X (ndarray): [N x P] Predictor values.

        Returns:
            p (ndarray): [N x K] Predicted probabilities, where p[i, k] is the
                probability that X[i] is class k.
        """
        return self._M * self._model.predict_probabilities(X)

    @property
    def labels(self) -> np.ndarray:
        """The label names associated with the model.

        Returns:
            labels (ndarray): [K] Label names of the data, such that labels[k]
                is the name of the kth label.
        """
        return self._model.labels
