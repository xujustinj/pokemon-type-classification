## `util/accuracy.py`

```{py}
from multiset import Multiset
import numpy as np
from sklearn.metrics import accuracy_score

def multiset_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Evaluate the accuracy of predictions versus test data.

    For multilabel predictions, the labels are treated as multisets, and the
    accuracy is measured as the multiset intersection between the true and
    predicted labels.

    Args:
        y_true (ndarray): [N] or [N x M] true class labels.
        y_pred (ndarray): [N] or [N x M] predicted class labels.

    Returns:
        accuracy (float): Accuracy (in the range [0,1]) of the predictions.
    """
    assert y_true.shape == y_pred.shape

    if len(y_true.shape) == 1:
        return float(accuracy_score(y_true=y_true, y_pred=y_pred))

    else:
        N, M = y_true.shape
        # the easiest way to do this is actually just a loop
        correct = np.array([
            len(Multiset(y_true[i,:]) & Multiset(y_pred[i,:])) / M
            for i in range(N)
        ])
        return correct.mean()
```
