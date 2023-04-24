import numpy as np


def duplicate(X: np.ndarray, y: np.ndarray):
    """Duplicates the given data.

    Specifically, observation i is duplicated M times, one for each of its
    labels, ending up at indices [i*M, (i+1)*M).

    Args:
        X (ndarray): [N x P] predictor values.
        y (ndarray): [N x M] labels.

    Returns:
        X (ndarray): [N*M x P] predictor values.
        y (ndarray): [N*M] labels.
    """
    N, P = X.shape
    assert y.shape[0] == N
    _, M = y.shape

    X = X.repeat(repeats=M, axis=0)
    assert X.shape == (N * M, P)

    y = y.flatten()
    assert y.shape == (N * M,)

    return X, y


def predict_multiset_indices(p: np.ndarray, cardinality: int):
    """Convert a list of probabilities to indices of a multiset prediction.

    Repeatedly predicts the highest label and subtracts 1 from its probability.

    Args:
        p (ndarray): [N x K] "probabilities", where p[i,k] is the expected
            multiplicity of label k in observation i.
        cardinality (int): The cardinality M of the predicted multiset (sum of
            element multiplicities).

    Returns:
        indices (ndarray): [N x M] label indices, where indices[i] contains the
            M labels (counting possible duplicates) predicted for observation i.
            Labels are ordered according to the algorithm (repeatedly predicting
            one label and subtracting from its probability).
    """
    N, K = p.shape

    all_ids = []
    for _ in range(cardinality):
        ids = p.argmax(axis=-1)
        for i, k in enumerate(ids):
            p[i,k] -= 1
        all_ids.append(ids)

    all_ids = np.stack(all_ids, axis=-1)
    assert all_ids.shape == (N, cardinality)

    return all_ids
