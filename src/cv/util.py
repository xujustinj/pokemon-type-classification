# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 22:47:28 2023

@author: zhang
"""
import os
from typing import Any, TypeVar

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model import Model, Config
from train import Trainer

M = TypeVar("M", bound=Model)

NROW = 1054
NCOL = 54
NFEAT = 74


def nested_cv(trainer: Trainer[M], model_configs: list[Config], folder, n_folds=8):
    NCONFIG = len(model_configs)

    # load training dataset
    poke_data = pd.read_csv("../data/scrape_from_scratch.csv")
    assert poke_data.shape == (NROW, NCOL)

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

    model_config_columns = list(model_configs[0].keys())
    RESULT_LISTS = []

    if not os.path.exists(folder):
        os.makedirs(folder)

    # for model_config in model_configs:
    cv_outer = StratifiedKFold(
        n_splits=n_folds,  # number of folds
        shuffle=True,  # protects against data being ordered, e.g., all successes first
        random_state=2023
    )

    for outer_idx, (train_id_outer, test_id_outer) in enumerate(cv_outer.split(X, y)):
        X_train_outer = X[train_id_outer, :]
        y_train_outer = y[train_id_outer]
        X_test_outer = X[test_id_outer, :]
        y_test_outer = y[test_id_outer]

        cv_inner = StratifiedKFold(
            n_splits=n_folds,  # number of folds
            shuffle=True,  # protects against data being ordered, e.g., all successes first
            random_state=2023
        )

        best_model_config = None
        all_metrics = []

        print(f"Outer CV: {outer_idx+1}")

        for inner_idx, (train_id_inner, test_id_inner) in enumerate(
            cv_inner.split(X_train_outer, y_train_outer)
        ):
            # foldwise training and test data
            X_train_inner = X_train_outer[train_id_inner, :]
            y_train_inner = y_train_outer[train_id_inner]
            X_test_inner = X_train_outer[test_id_inner, :]
            y_test_inner = y_train_outer[test_id_inner]

            print(f"  Inner CV: {inner_idx+1}")

            fold_metrics = []
            for k, model_config in enumerate(model_configs):
                model = trainer.train(
                    X_train=X_train_inner,
                    y_train=y_train_inner,
                    config=model_config,
                )
                score = model.evaluate(X_test_inner, y_test_inner)
                print(f"    {k+1}/{NCONFIG}\t{score:06f}\t{model_config}")
                fold_metrics.append(score)

            fold_metrics = np.array(fold_metrics)
            all_metrics.append(fold_metrics)

        all_metrics = np.array(all_metrics)
        average_inner_cv_error = np.average(all_metrics, axis=0)
        best_model_config = model_configs[np.argmax(average_inner_cv_error)]

        model = trainer.train(
            X_train=X_train_outer,
            y_train=y_train_outer,
            config=best_model_config)
        model.save(os.path.join(folder, f"cv-{outer_idx+1}.mdl"))
        acc_score = model.evaluate(X_test_outer, y_test_outer)

        RESULT_LIST = []
        for k in model_config_columns:
            RESULT_LIST.append(best_model_config[k])
        RESULT_LIST.append(acc_score)
        RESULT_LISTS.append(RESULT_LIST)

    model_config_columns.append("acc_scores")
    result_df = pd.DataFrame(RESULT_LISTS, columns=model_config_columns)
    result_df.to_csv(os.path.join(folder, "result.csv"))
    print(np.average(result_df['acc_scores']))
