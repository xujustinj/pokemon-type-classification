import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_DATA_PATH = "../data/scrape_from_scratch.csv"
_NROW = 1054
_NCOL = 54
_NFEAT = 74


def load_data() -> tuple[
    np.ndarray[(_NROW, _NFEAT), float],
    np.ndarray[(_NROW,), object]
]:
    """
    Load the dataset.
    Apply normalization and one-hot encoding transformations.

    Returns:
        X: the predictors (encoded as floating point numbers)
        y: the labels (encoded as strings)
    """
    poke_data = pd.read_csv(_DATA_PATH)
    assert poke_data.shape == (_NROW, _NCOL)

    ct = ColumnTransformer(
        [
            (
                "normalize",
                StandardScaler(),
                make_column_selector(dtype_include=np.number)  # type: ignore
            ),
            (
                "onehot",
                OneHotEncoder(),
                make_column_selector(dtype_include=object),  # type: ignore
            ),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
    )

    X = ct.fit_transform(poke_data.drop(columns=['type_1']))
    assert isinstance(X, np.ndarray)
    assert X.shape == (_NROW, _NFEAT)
    assert X.dtype == float

    y = poke_data['type_1'].to_numpy()
    assert isinstance(y, np.ndarray)
    assert y.shape == (_NROW,)
    assert y.dtype == object

    return X, y
