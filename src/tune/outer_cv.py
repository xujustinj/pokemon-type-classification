import os
from time import time
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from model import Model
from util import load_data, multiset_accuracy, plot_confusion
from .tuner import Tuner, SearchSpace


M = TypeVar("M", bound=Model)

BASE_MODEL_DIR = "../models/"


def outer_cv(
    tuner: Tuner[M],
    search: SearchSpace,
    name: str,
    duplicate: bool,
    hard_mode: bool,
    n_folds_outer: int = 5,
    n_folds_inner: int = 5,
) -> float:
    """Apply nested cross validation and find average cross validation accuracy.

    Apply nested cross validation and find average cross validation accuracy
    to one model type with hyperparamater tuning in inner folds.

    Args:
        tuner (Trainer[M]): The trainer for model type M
        search (SearchSpace): The search space for hyperparameters
        name (str): The name of the model
        n_folds_outer (int): The number of outer folds
        n_folds_inner (int): The number of inner folds
        hard_mode (bool): False if including all columns, True if removing against and type 2 columns

    Returns:
        float: The outer fold average cross validation accuracy
    """
    X, y = load_data(hard_mode=hard_mode, duplicate=duplicate)
    results: list[dict[str, Any]] = []

    if duplicate:
        if hard_mode:
            name = f"{name} (Duplication, Hard)"
        else:
            name = f"{name} (Duplication)"
    else:
        if hard_mode:
            name = f"{name} (Hard)"
        else:
            name = f"{name} (Base)"

    model_dir = os.path.join(
        BASE_MODEL_DIR,
        name.replace(" ", "_") \
            .replace("(", "").replace(")", "") \
            .replace(",", "") \
            .lower(),
    )
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    outer_splitter = (KFold if duplicate else StratifiedKFold)(
        n_splits=n_folds_outer,  # number of folds
        shuffle=True,  # protects against data being ordered, e.g., all successes first
        random_state=441,
    )

    if duplicate:
        y_merged = np.array(["/".join(sorted(row)) for row in y])
        outer_splits = list(outer_splitter.split(X, y_merged))
    else:
        outer_splits = list(outer_splitter.split(X, y))

    predictions = y.copy()
    for i, (train_ids, test_ids) in enumerate(outer_splits):
        id = i+1
        model_path = os.path.join(model_dir, f"cv-{id}.mdl")

        if os.path.exists(model_path):
            print(f"{id}  Cached result loaded from {model_path}")
            model: M = Model.load(model_path)
        else:
            X_train: np.ndarray = X[train_ids, :]
            y_train: np.ndarray = y[train_ids]

            split = (KFold if duplicate else StratifiedKFold)(
                n_splits=n_folds_inner,
                shuffle=True,
                random_state=441,
            )

            start_s = time()
            model = tuner.tune(
                X_train=X_train,
                y_train=y_train,
                search=search,
                split=split,
            )
            end_s = time()
            elapsed_s = end_s - start_s
            print(f"{id}  Tuned model in {elapsed_s} seconds")

            model.save(model_path)
            print(f"{id}  Saved to cache at {model_path}")

        print(f"{id}  Best configuration: {model.config}")

        X_test: np.ndarray = X[test_ids, :]
        y_test: np.ndarray = y[test_ids]
        y_pred = model.predict(X_test)
        accuracy = multiset_accuracy(y_test, y_pred)
        print(f"{id}  Accuracy: {accuracy}")

        results.append(dict(**model.config, accuracy=accuracy))
        predictions[test_ids] = y_pred

    labels = model.labels

    plot_confusion(
        y_true=y,
        y_pred=predictions,
        labels=labels,
        normalize="true",
        name=f"{name} by true label",
        path=os.path.join(model_dir, "confusion_matrix_by_true.png"),
    )

    plot_confusion(
        y_true=y,
        y_pred=predictions,
        labels=labels,
        normalize="pred",
        name=f"{name} by predicted label",
        path=os.path.join(model_dir, "confusion_matrix_by_pred.png"),
    )

    results_df = pd.DataFrame.from_records(results)
    results_df.to_csv(os.path.join(model_dir, "result.csv"))
    return np.average(results_df['accuracy'])
