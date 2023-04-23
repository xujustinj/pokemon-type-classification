from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def multiset_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    normalize: Literal["true", "pred", "all"] | None
) -> np.ndarray:
    """Generate a confusion matrix for M-label multiset classification.

    In single-label prediction, the (raw) confusion matrix can be interpreted as
    a sum of outer products between one-hot vectors encoding the true and
    predicted classes.

    Here we treat the multi-label equivalent of the confusion matrix as also a
    sum of outer products, but the encodings are no longer one-hot.

    Args:
        y_true (ndarray): [N x M] true class labels.
        y_pred (ndarray): [N x M] predicted class labels.
        labels (list[str]): All possible labels.
        normalize ("true" | "pred" | "all" | None): Normalizes the confusion
            matrix over the true class (rows), predicted class (columns), or
            the population. If None, the confusion matrix will not be
            normalized. "Normalizing" in this context means scaling such that
            the resulting sum is M (not 1). The reason for preferring M is so
            that the normalized entries may be interpreted as mean cardinality.
    Returns:
        cm (ndarray): [K x K] matrix where cm[i,j] is the count/proportion
            associated with true class i and predicted class j.
    """
    assert y_true.shape == y_pred.shape
    N, M = y_true.shape
    K = len(labels)

    label_to_index = {label: i for i, label in enumerate(labels)}

    cm = np.zeros((K, K), dtype=float)
    for _ in range(N):
        ind_true = [label_to_index[label] for label in y_true[i,:]]
        ind_pred = [label_to_index[label] for label in y_pred[i,:]]
        for i in ind_true:
            for j in ind_pred:
                cm[i, j] += 1.0

    # copied right out of sklearn's confusion matrix code
    if normalize == "true":
        cm = M * cm / cm.sum(axis=1, keepdims=True)
    elif normalize == "pred":
        cm = M * cm / cm.sum(axis=0, keepdims=True)
    elif normalize == "all":
        cm = M * cm / cm.sum()
    cm = np.nan_to_num(cm)

    return cm


def confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    normalize: Literal["true", "pred", "all"] | None
) -> np.ndarray:
    """Generate a confusion matrix for M-label multiset classification.

    For details, see sklearn's documentation of confusion_matrix, and
    multiset_confusion above.
    """
    if len(y_true.shape) == 1:
        return confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            normalize=normalize,
        )
    else:
        assert len(y_true.shape) == 2
        return multiset_confusion(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            normalize=normalize,
        )



def plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    normalize: Literal["true", "pred", "all"] | None,
    name: str,
    path: str,
) -> None:
    """Plot, display, and save a confusion matrix.

    Args:
        cm (ndarray): [K x K] confusion matrix.
    """
    cm = confusion(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        normalize=normalize,
    )
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.figure(figsize=(10,10))
    display.plot(xticks_rotation='vertical', values_format=".0%")
    plt.title(f"Confusion Matrix for {name}")
    plt.savefig(path)
    plt.show()
