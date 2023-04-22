import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_DATA_PATH = "../data/scrape_from_scratch.csv"
_NROW = 1054
_NCOL = 54
_NFEAT = 74
_NCLASS = 18


def load_data(duplicate: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Load the dataset.

    Apply normalization and one-hot encoding transformations.

    Args:
        duplicate (bool): Whether to apply the duplication method. If applied,
            the duplicated data will be interwoven so that all duplicated
            samples corresponding to the same original sample are together, but
            they themselves will be in random order.

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

    if duplicate:
        is_pure = poke_data["type_2"] == "None"
        poke_data.loc[is_pure, "type_2"] = poke_data.loc[is_pure, "type_1"]

        poke_data_1 = poke_data.rename(columns=dict(type_1="type")).drop(columns="type_2")
        poke_data_2 = poke_data.rename(columns=dict(type_2="type")).drop(columns="type_1")
        poke_data = pd.concat([poke_data_1, poke_data_2])

        # interweave: shuffle, resort, reindex
        poke_data = poke_data.sample(frac=1, random_state=441)
        # sort must be stable, such as merge sort
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_index.html
        poke_data.sort_index(kind="mergesort", inplace=True)
        poke_data.reset_index(drop=True, inplace=True)

        assert poke_data.shape == (2 * _NROW, _NCOL - 1)

        X = ct.fit_transform(poke_data.drop(columns="type"))
        assert isinstance(X, np.ndarray)
        assert X.shape == (2 * _NROW, _NFEAT - (_NCLASS+1))  # -1 because None type
        assert X.dtype == float

        y = poke_data["type"]
        assert len(y.unique()) == _NCLASS
        y = y.to_numpy()
        assert isinstance(y, np.ndarray)
        assert y.shape == (2 * _NROW,)
        assert y.dtype == object

    else:
        X = ct.fit_transform(poke_data.drop(columns="type_1"))
        assert isinstance(X, np.ndarray)
        assert X.shape == (_NROW, _NFEAT)
        assert X.dtype == float

        y = poke_data["type_1"]
        assert len(y.unique()) == _NCLASS
        y = y.to_numpy()
        assert isinstance(y, np.ndarray)
        assert y.shape == (_NROW,)
        assert y.dtype == object

    return X, y
