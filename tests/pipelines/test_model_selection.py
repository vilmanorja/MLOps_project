import numpy as np
import pandas as pd
import pytest
import pickle
import warnings
import mlflow,yaml
import logging
import os
import sys 
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore", category=Warning)
full_path = os.getcwd()
sys.path.append(os.path.abspath("src"))

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from mlops_credit_scoring.pipelines.model_selection.nodes import model_selection

logger = logging.getLogger(__name__)

@pytest.mark.slow
def test_model_selection():
    """
    Test that the model selection node returns a model with a score
    """
    # read sample data
    X_train = pd.read_csv(full_path + "/tests/pipelines/sample/X_train_processed.csv") 
    X_test = pd.read_csv(full_path + "/tests/pipelines/sample/X_train_processed.csv") 
    y_train = pd.read_csv(full_path + "/tests/pipelines/sample/y_train_processed.csv") 
    y_test = pd.read_csv(full_path + "/tests/pipelines/sample/y_train_processed.csv") 
    
    champion_dict = {'classifier': None, 'test_f1': 0}

    champion_model = None
    
    parameters = {
        'hyperparameters': {
            'RandomForestClassifier': {'n_estimators': [ 200, 300], 'random_state': [42], 'class_weight': ['balanced']},
            'GradientBoostingClassifier': {'n_estimators': [300, 500], 'random_state': [42]},
            'LogisticRegression': {'C': [0.01, 0.1, 1.0, 10.0], 'solver': ['liblinear'], 'random_state': [42], 'class_weight': ['balanced']},
            'XGBClassifier': {'n_estimators': [300, 500], 'random_state': [42]}
        },
        'use_feature_selection': True
    }

    # Open the pickle file in binary read mode
    with open(full_path + "/data/06_models/best_cols.pkl", 'rb') as f:
        best_columns = pickle.load(f)

    # Run the model selection node
    model = model_selection(X_train, X_test, y_train, y_test, champion_dict, champion_model, parameters, best_columns)
    
    # Check that the returned value is a dictionary
    assert isinstance(model, RandomForestClassifier) or isinstance(model, GradientBoostingClassifier) or isinstance(model, LogisticRegression) or isinstance(model, XGBClassifier)

# print('Test model selection')
# test_model_selection()
# print('Model selection succeeded')