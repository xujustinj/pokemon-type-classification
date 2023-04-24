import logging
import multiprocessing as mp
import os
from time import time
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from model import Model
from util import load_data, multiset_accuracy, plot_confusion
from .duplication import DuplicationTuner
from .tuner import Tuner, SearchSpace


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
    """Apply nested cross validation and find average cross validation accuracy. 

    Apply nested cross validation and find average cross validation accuracy 
    to one model type with hyperparamater tuning in inner folds. 

    Args:
        tuner (Trainer[M]): The trainer for model type M
        search (SearchSpace): The search space for hyperparameters
        n_folds_inner (int): The number of inner folds
        model_dir (str): The path to save trained models
        X (np.ndarray): The X-values of the data
        y (np.ndarray): The y-values of the data
        id (int): The index of the outer fold and trained model
        train_ids (np.ndarray): The indices of training data
        test_ids (np.ndarray): The indices of testing data

    Returns:
        tuple[M, float, np.ndarray]: tuple containing trained model of type M, 
                                     testing accuracy of the model in this fold, 
                                     and the prediction result on test set
    """
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
    y_pred = model.predict(X_test)
    accuracy = multiset_accuracy(y_test, y_pred)
    _log.info(f"{id}  Accuracy: {accuracy}")

    return model, accuracy, y_pred


def outer_cv(
    tuner: Tuner[M],
    search: SearchSpace,
    name: str,
    n_folds_outer: int = 5,
    n_folds_inner: int = 5,
    n_jobs: int = 1,
    hard_mode: bool = False,
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
        n_jobs (int): The number of jobs to run for multiprocessing
        hard_mode (bool): False if including all columns, True if removing against and type 2 columns

    Returns:
        float: The outer fold average cross validation accuracy
    """
    duplicate = isinstance(tuner, DuplicationTuner)

    n_jobs = min(n_jobs, n_folds_outer)  # cannot exceed one job per outer fold

    X, y = load_data(hard_mode=hard_mode, duplicate=duplicate)
    results: list[dict[str, Any]] = []

    model_dir = os.path.join(
        BASE_MODEL_DIR,
        name.replace(" ", "_") \
            .replace("(", "") \
            .replace(")", "") \
            .lower(),
    )
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # for model_config in model_configs:
    outer_splitter = StratifiedKFold(
        n_splits=n_folds_outer,  # number of folds
        shuffle=True,  # protects against data being ordered, e.g., all successes first
        random_state=441,
    )

    if duplicate:
        y_merged = np.array(["/".join(sorted(row)) for row in y])
        outer_splits = list(outer_splitter.split(X, y_merged))
    else:
        outer_splits = list(outer_splitter.split(X, y))

    if n_jobs > 1:
        print(f"Outer CV using {n_jobs} cores")
        with mp.Pool(n_jobs) as pool:
            models = pool.starmap(_outer_cv_fold, [
                (tuner, search, n_folds_inner, model_dir, X, y, i+1, train_ids, test_ids)
                for i, (train_ids, test_ids) in enumerate(outer_splits)
            ])
    else:
        models = [
            _outer_cv_fold(
                tuner=tuner,
                search=search,
                n_folds_inner=n_folds_inner,
                model_dir=model_dir,
                X=X,
                y=y,
                id=i+1,
                train_ids=train_ids,
                test_ids=test_ids
            )
            for i, (train_ids, test_ids) in enumerate(outer_splits)
        ]

    y_pred = y.copy()
    for (model, accuracy, predictions), (_, test_ids) in zip(models, outer_splits):
        results.append(dict(**model.config, accuracy=accuracy))
        y_pred[test_ids] = predictions

    labels = models[0][0].labels

    plot_confusion(
        y_true=y,
        y_pred=y_pred,
        labels=labels,
        normalize="true",
        name=f"{name} by true label",
        path=os.path.join(model_dir, "confusion_matrix_by_true.png"),
    )

    plot_confusion(
        y_true=y,
        y_pred=y_pred,
        labels=labels,
        normalize="pred",
        name=f"{name} by predicted label",
        path=os.path.join(model_dir, "confusion_matrix_by_pred.png"),
    )

    results_df = pd.DataFrame.from_records(results)
    results_df.to_csv(os.path.join(model_dir, "result.csv"))
    return np.average(results_df['accuracy'])
