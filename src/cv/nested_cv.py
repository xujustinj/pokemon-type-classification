# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 22:47:28 2023

@author: zhang
"""
import os
from typing import Any, TypeVar

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from model import Model, Config
from train import Trainer
from util import load_data

M = TypeVar("M", bound=Model)


def nested_cv(trainer: Trainer[M], model_configs: list[Config], folder, name, n_folds=8):
    """Apply nested cross validation and find average cross validation accuracy. 

    Apply nested cross validation and find average cross validation accuracy 
    to one model type with hyperparamater tuning in inner folds. 

    Args:
        trainer (Trainer[M]): trainer for model type M
        model_configs (list[Config]): list of sets of model hyperparameters
        folder (str): the folder path used to store results
        name (str): model name
        n_folds(int): number of folds

    Returns:
        Iterator[dict[str, Any]]: sets of hyperparameters

    """
    X, y = load_data()

    NCONFIG = len(model_configs)
    model_config_columns = list(model_configs[0].keys())
    RESULT_LISTS = []
    CONFUSION_PRED = []
    CONFUSION_TRUE = []
    CLASSES = []

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
        pred = model.predict(X_test_outer)

        RESULT_LIST = []
        for k in model_config_columns:
            RESULT_LIST.append(best_model_config[k])
        RESULT_LIST.append(acc_score)
        RESULT_LISTS.append(RESULT_LIST)

        CONFUSION_PRED.extend(pred)
        CONFUSION_TRUE.extend(y_test_outer)
        if len(CLASSES) == 0:
            CLASSES=model.labels()

    cm = confusion_matrix(CONFUSION_TRUE, CONFUSION_PRED, labels=CLASSES)
    display = ConfusionMatrixDisplay(confusion_matrix=cm,
                                     display_labels=CLASSES)

    plt.rcParams['figure.figsize'] = [10, 10]
    plt.figure(figsize=(10,10))
    display.plot(xticks_rotation='vertical')
    plt.title(f"Confusion Matrix on Test Data for {name}")
    plt.savefig(os.path.join(folder, "conf_matrix.png"))
    plt.show()
    model_config_columns.append("acc_scores")
    result_df = pd.DataFrame(RESULT_LISTS, columns=model_config_columns)
    result_df.to_csv(os.path.join(folder, "result.csv"))
    print(np.average(result_df['acc_scores']))
