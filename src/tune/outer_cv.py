import os
from typing import Any, TypeVar

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model import Model, Config
from .tuner import Tuner, SearchSpace, Splitter

M = TypeVar("M", bound=Model)

NROW = 1054
NCOL = 54
NFEAT = 74

DATA_PATH = "../data/scrape_from_scratch.csv"
BASE_MODEL_DIR = "../models/"


def outer_cv(
    tuner: Tuner[M],
    search: SearchSpace,
    name,
    n_folds_outer=5,
    n_folds_inner=5,
):
    # load training dataset
    poke_data = pd.read_csv(DATA_PATH)
    assert poke_data.shape == (NROW, NCOL)

    MODEL_DIR = os.path.join(BASE_MODEL_DIR, name)

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
    assert X.shape == (NROW, NFEAT)

    y: np.ndarray = poke_data['type_1'].to_numpy()
    assert y.shape == (NROW,)

    configs: list[Config] = []

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # for model_config in model_configs:
    outer_splitter = StratifiedKFold(
        n_splits=n_folds_outer,  # number of folds
        shuffle=True,  # protects against data being ordered, e.g., all successes first
        random_state=441,
    )

    inner_splitter = StratifiedKFold(
        n_splits=n_folds_inner,
        shuffle=True,
        random_state=441,
    )

    for outer_idx, (train_id_outer, test_id_outer) in enumerate(outer_splitter.split(X, y)):
        X_train_outer = X[train_id_outer, :]
        y_train_outer = y[train_id_outer]
        X_test_outer = X[test_id_outer, :]
        y_test_outer = y[test_id_outer]

        print(f"Outer CV: {outer_idx+1}")

        model = tuner.tune(
            X_train=X_train_outer,
            y_train=y_train_outer,
            search=search,
            split=inner_splitter,
        )
        model.save(os.path.join(MODEL_DIR, f"cv-{outer_idx+1}.mdl"))
        accuracy = model.evaluate(X_test_outer, y_test_outer)
        configs.append(dict(**model.config, accuracy=accuracy))

    result_df = pd.DataFrame.from_records(configs)
    result_df.to_csv(os.path.join(MODEL_DIR, "result.csv"))
    return np.average(result_df['accuracy'])
