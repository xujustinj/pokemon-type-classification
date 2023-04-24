## `util/data.py`

```{py}
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

_DATA_PATH = "../data/scraped.csv"

_N_CLASS = 18
_N_TYPE_2 = _N_CLASS + 1
_N_STATUS = 4


def load_data(
    duplicate: bool = False,
    hard_mode: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the dataset with normalization and one-hot encoding transformations.

    Args:
        duplicate (bool): Whether to apply the duplication method. If so, move
            type_2 from the predictors X to the target labels y.
        hard_mode (bool): Whether to exclude the against_from_* and type_2
            predictors, which make the classification task markedly easier.

    Returns:
        X (ndarray): The predictors (encoded as floating point numbers)
        y (ndarray): The labels (encoded as strings)
    """
    n_row = 1054
    n_col = 54

    poke_data = pd.read_csv(_DATA_PATH)
    assert poke_data.shape == (n_row, n_col)

    if hard_mode:
        against_from_columns = poke_data.columns[poke_data.columns.str.contains("damage_from_")]
        poke_data.drop(columns=against_from_columns, inplace=True)
        n_col = n_col - _N_CLASS

        if not duplicate:
            poke_data.drop(columns=["type_2"], inplace=True)
            n_col = n_col - 1

    assert poke_data.shape == (n_row, n_col)

    ct = ColumnTransformer(
        [
            (
                "normalize",
                MinMaxScaler(),
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
        # replace None type_2 with another copy of type_1
        is_pure = poke_data["type_2"] == "None"
        poke_data.loc[is_pure, "type_2"] = poke_data.loc[is_pure, "type_1"]

        X = ct.fit_transform(poke_data.drop(columns=["type_1", "type_2"]))
        n_feat = n_col - 2  # remove types
        n_feat = n_feat + (_N_STATUS - 1)  # add one-hot encoding of status
        assert isinstance(X, np.ndarray)
        assert X.shape == (n_row, n_feat)  # -1 because None type
        assert X.dtype == float

        y = poke_data[["type_1", "type_2"]]
        assert len(pd.concat([y["type_1"], y["type_2"]]).unique()) == _N_CLASS
        y = y.to_numpy()
        assert isinstance(y, np.ndarray)
        assert y.shape == (n_row, 2)
        assert y.dtype == object

    else:
        X = ct.fit_transform(poke_data.drop(columns="type_1"))
        n_feat = n_col - 1  # remove type_1
        n_feat = n_feat + (_N_STATUS - 1)  # add one-hot encoding of status
        if not hard_mode:
            n_feat = n_feat + (_N_TYPE_2 - 1)  # add one-hot encoding of type_2
        assert isinstance(X, np.ndarray)
        assert X.shape == (n_row, n_feat)
        assert X.dtype == float

        y = poke_data["type_1"]
        assert len(y.unique()) == _N_CLASS
        y = y.to_numpy()
        assert isinstance(y, np.ndarray)
        assert y.shape == (n_row,)
        assert y.dtype == object

    return X, y
```
