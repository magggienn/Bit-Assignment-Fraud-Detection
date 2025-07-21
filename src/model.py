""" This module contains functions for training machine learning models, optimizing hyperparameters using Optuna, and evaluating model performance.
It includes functions for training XGBoost models, evaluating them, and performing logistic regression benchmarks.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report 
import xgboost as xgb
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def split_data(X,y):
    """ Split the data into train, validation, and test sets """

    # Train, validation, test sets
    #   60%,     20%,    20% split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

def compute_scale_pos_weight(y):
    """Compute scale_pos_weight from binary labels"""
    num_neg = (y == 0).sum()
    num_pos = (y == 1).sum()
    return num_neg / num_pos if num_pos != 0 else 1

def objective(trial, X_train, X_val, X_test, y_train, y_val, y_test):
    """ Optimize the data for training """
    
    # define the hyperparameters to optimize
    param = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
         "scale_pos_weight": compute_scale_pos_weight(y_train),
        "use_label_encoder": False,
        "random_state": 42,
        "deterministic_histogram": True,  # to ensure reproducibility 100%
        "tree_method": "exact"
    }

    model = xgb.XGBClassifier(**param)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    # predict on the validation set
    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    return -auc 
    
def train_model(best_params, X_train, X_val, X_test, y_train, y_val):
    """ Train a model on the given data """

    # initialize the model with the best hyperparameters
    model = xgb.XGBClassifier(**best_params, use_label_encoder=False, objective='binary:logistic', eval_metric='auc', random_state=42)
    model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = model.predict(X_test)
    return model, y_test_pred, y_test_pred_proba

def evaluate_model(y_test, y_test_pred, y_test_pred_proba):
    """ Evaluate the model performance """
    print(classification_report(y_test, y_test_pred))
    auc = roc_auc_score(y_test, y_test_pred_proba)
    print(f'Test ROC-AUC: {auc}')
    return {'roc_auc': auc}

def logistic_regression_benchmark(X_train, X_test, y_train, y_test):
    """ Train a Logistic Regression model as a benchmark """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred_log = model.predict(X_test_scaled)
    y_pred_proba_log = model.predict_proba(X_test_scaled)[:, 1]
    print(classification_report(y_test, y_pred_log))
    print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba_log))
    return model, y_pred_log, y_pred_proba_log