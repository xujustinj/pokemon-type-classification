import numpy as np

from model import DuplicationModel
from .tuner import Tuner, SearchSpace, Splitter


def _duplicate(ids: np.ndarray, M: int):
    """Convert a set of indices on the original data to an equivalent set of
    indices for the duplicated set.
    """
    return np.array([i * M + j for j in range(M) for i in ids])


class DuplicationTuner(Tuner[DuplicationModel]):
    """Tunes an M-label duplication model using another tuner.

    Args:
        tuner (Tuner): The tuner for optimizing the single-label classifier.
    """
    def __init__(self, tuner: Tuner):
        self._tuner = tuner

    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        search: SearchSpace,
        split: Splitter,
    ) -> DuplicationModel:
        _, M = y_train.shape

        # index M*i+j corresponds to the jth label of the ith example
        dup_X_train = X_train.repeat(repeats=M, axis=0)
        dup_y_train = y_train.flatten()
        y_train_merged = np.array(["/".join(sorted(row)) for row in y_train])
        dup_split = (
            (_duplicate(train_ids, M), _duplicate(test_ids, M))
            for train_ids, test_ids in split.split(X_train, y_train_merged)
        )

        model = self._tuner.tune(
            X_train=dup_X_train,
            y_train=dup_y_train,
            search=search,
            split=dup_split,
        )
        return DuplicationModel(model=model, M=M)
