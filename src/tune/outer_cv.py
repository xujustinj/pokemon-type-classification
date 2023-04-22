import logging
import multiprocessing as mp
import os
from time import time
from typing import Any, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold

from model import Model
from util import load_data
from .tuner import Tuner, SearchSpace, Splitter


M = TypeVar("M", bound=Model)

BASE_MODEL_DIR = "../models/"


# MULTIPROCESSING

_CPUS = len(os.sched_getaffinity(0))
# use one fewer core to avoid hogging all resources
_PARALLELISM = max(_CPUS - 1, 1)

_log = mp.log_to_stderr()
_log.setLevel(logging.INFO)


def _outer_cv_fold(
    tuner: Tuner[M],
    search: SearchSpace,
    n_folds_inner: int,
    model_dir: str,
    X: np.ndarray,
    y: np.ndarray,
    id: int,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
) -> tuple[M, float, np.ndarray]:
    model_path = os.path.join(model_dir, f"cv-{id}.mdl")

    if os.path.exists(model_path):
        _log.info(f"{id}  Cached result loaded from {model_path}")
        model: M = Model.load(model_path)
    else:
        X_train: np.ndarray = X[train_ids, :]
        y_train: np.ndarray = y[train_ids]

        split = StratifiedKFold(
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
        _log.info(f"{id}  Tuned model in {elapsed_s} seconds")

        model.save(model_path)
        _log.info(f"{id}  Saved to cache at {model_path}")

    _log.info(f"{id}  Best configuration: {model.config}")

    X_test: np.ndarray = X[test_ids, :]
    y_test: np.ndarray = y[test_ids]
    accuracy = model.evaluate(X_test, y_test)
    _log.info(f"{id}\tAccuracy: {accuracy}")

    predictions = model.predict(X_test)

    return model, accuracy, predictions


def outer_cv(
    tuner: Tuner[M],
    search: SearchSpace,
    name: str,
    n_folds_outer: int = 5,
    n_folds_inner: int = 5,
    parallelism: int = _PARALLELISM,
) -> float:
    parallelism = min(parallelism, n_folds_outer)
    print(f"Outer CV using {parallelism} cores")

    X, y = load_data()
    results: list[dict[str, Any]] = []

    model_dir = os.path.join(BASE_MODEL_DIR, name.replace(" ", "_").lower())
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # for model_config in model_configs:
    outer_splitter = StratifiedKFold(
        n_splits=n_folds_outer,  # number of folds
        shuffle=True,  # protects against data being ordered, e.g., all successes first
        random_state=441,
    )

    outer_splits = list(outer_splitter.split(X, y))
    with mp.Pool(parallelism) as pool:
        pool_results = pool.starmap(_outer_cv_fold, [
            (tuner, search, n_folds_inner, model_dir, X, y, i+1, train_ids, test_ids)
            for i, (train_ids, test_ids) in enumerate(outer_splits)
        ])

    y_pred = y.copy()
    for (model, accuracy, predictions), (_, test_ids) in zip(pool_results, outer_splits):
        results.append(dict(**model.config, accuracy=accuracy))
        y_pred[test_ids] = predictions

    labels = pool_results[0][0].labels()
    cm = confusion_matrix(y_true=y, y_pred=y_pred, labels=labels)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.figure(figsize=(10,10))
    display.plot(xticks_rotation='vertical')
    plt.title(f"Confusion Matrix on Test Data for {name}")
    plt.savefig(os.path.join(model_dir, "confusion_matrix.png"))
    plt.show()

    results_df = pd.DataFrame.from_records(results)
    results_df.to_csv(os.path.join(model_dir, "result.csv"))
    return np.average(results_df['accuracy'])
