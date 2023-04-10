# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 22:47:28 2023

@author: zhang
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder

def nested_cv(train_model_func, model_type, model_configs, folder, n_folds=8): 
    # load training dataset
    poke_data = pd.read_csv("../data/2-generate_features.csv")
    poke_data = poke_data.drop(columns=['has_gender'])
    poke_data = poke_data.drop(columns=['type_2'])
    print(poke_data.shape)
    ct = ColumnTransformer([
    ('onehotstatus',
        OneHotEncoder(),
        make_column_selector(pattern='status'))],
    remainder='passthrough',
    verbose_feature_names_out=False)
    
    X = ct.fit_transform(poke_data.drop(columns=['type_1']))
    y = poke_data['type_1'].to_numpy()
    print(X.shape)
    print(y.shape)
    model_config_columns = list(model_configs[0].keys())
    RESULT_LISTS = []
    
    if not os.path.exists(folder): 
        os.makedirs(folder)
    i = 0
    
    # for model_config in model_configs: 
    cv_outer = StratifiedKFold(
        n_splits=n_folds,  # number of folds
        shuffle=True,  # protects against data being ordered, e.g., all successes first
        random_state=2023
    )
    
    for (train_id_outer, test_id_outer) in cv_outer.split(X, y):
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
        
        print("Outer CV run: " + str(i))
        
        
        for (train_id_inner, test_id_inner) in cv_inner.split(X_train_outer, y_train_outer):
            # foldwise training and test data
            X_train_inner = X_train_outer[train_id_inner, :]
            y_train_inner = y_train_outer[train_id_inner]
            X_test_inner = X_train_outer[test_id_inner, :]
            y_test_inner = y_train_outer[test_id_inner]
            
            print("Inner CV run")
            
            fold_metrics = []
            for model_config in model_configs: 
                print("Building Model")
                model = train_model_func(X_train_inner, y_train_inner, model_config)
                predictions = model.predict(X_test_inner)
                fold_metrics.append(model.evaluate(predictions, y_test_inner))
            
            fold_metrics = np.array(fold_metrics)
            
        all_metrics.append(fold_metrics)
        
        all_metrics = np.array(all_metrics)
        average_inner_cv_error = np.average(all_metrics, axis=0)
        best_model_config = model_configs[np.argmax(average_inner_cv_error)]
        
        model = train_model_func(X_train_outer, y_train_outer, best_model_config)
        model.save(folder + "cv" + str(i) + ".mdl")
        i += 1
        predictions = model.predict(X_test_outer)
        acc_score = model.evaluate(predictions, y_test_outer)
        
        RESULT_LIST = []
        for k in model_config_columns: 
            RESULT_LIST.append(best_model_config[k])
        RESULT_LIST.append(acc_score)
        RESULT_LISTS.append(RESULT_LIST)
    
    model_config_columns.append("acc_scores")
    result_df = pd.DataFrame(RESULT_LISTS, columns=model_config_columns)
    result_df.to_csv(folder + "result.csv")
    print(np.average(result_df['acc_scores']))

