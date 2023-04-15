from typing import Literal

from sklearn import linear_model
import numpy as np

from model import Config, LRModel
from .trainer import Trainer


MultiClass = Literal["auto", "ovr", "multinomial"]
Penalty = Literal["l1", "l2", "elasticnet"] | None


class LRTrainer(Trainer[LRModel]):
    def __init__(self):
        super().__init__()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        config: Config,
    ) -> LRModel:
        multi_class: MultiClass = config.get("multi_class", "auto")
        penalty: Penalty = config.get("penalty")
        C: float = config.get("C", 1.0)

        model = linear_model.LogisticRegression(
            multi_class=multi_class,
            penalty=penalty,
            C=C,
            max_iter=9001,  # practically unlimited iterations
            random_state=441,
        )

        model.fit(X_train, y_train)

        return LRModel(model, config)
