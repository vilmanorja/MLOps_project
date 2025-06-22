
import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np  
import yaml
import pickle
import warnings
warnings.filterwarnings("ignore", category=Warning)


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

import mlflow

logger = logging.getLogger(__name__)

def _get_or_create_experiment_id(experiment_name: str) -> str:
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        logger.info(f"Experiment '{experiment_name}' not found. Creating new one.")
        return mlflow.create_experiment(experiment_name)
    return exp.experiment_id
     
def model_selection(X_train: pd.DataFrame, 
                    X_test: pd.DataFrame, 
                    y_train: pd.DataFrame, 
                    y_test: pd.DataFrame,
                    champion_dict: Dict[str, Any],
                    champion_model : pickle.Pickler,
                    parameters: Dict[str, Any]):
    y_train = np.ravel(y_train)
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos
    models_dict = {
        'LogisticRegression': LogisticRegression(),
        'RandomForestClassifier': RandomForestClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        'XGBClassifier': XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42,scale_pos_weight=scale_pos_weight)
    }


    initial_results = {}   


    with open('conf/local/mlflow.yml') as f:
        experiment_name = yaml.load(f, Loader=yaml.loader.SafeLoader)['tracking']['experiment']['name']
        experiment_id = _get_or_create_experiment_id(experiment_name)
        logger.info(experiment_id)


        logger.info(' Step 1: Comparing models using 5-fold CV F1-score...')

    for model_name, model in models_dict.items():
        with mlflow.start_run(experiment_id=experiment_id,nested=True):
            mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)
      
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
            mean_f1 = np.mean(cv_scores)
            initial_results[model_name] = mean_f1

            run_id = mlflow.last_active_run().info.run_id
            logger.info(f" {model_name} mean CV F1-score: {mean_f1:.4f}")
            logger.info(f"Logged model : {model_name} in run {run_id}")
    
    best_model_name = max(initial_results, key=initial_results.get)
    best_model = models_dict[best_model_name]

    logger.info(f"Best model is {best_model_name} with score {initial_results[best_model_name]}")
    logger.info('Step 2: Hyperparameter tuning with 5-fold CV...')

    # Perform hyperparameter tuning with GridSearchCV
    param_grid = parameters['hyperparameters'][best_model_name]
    with mlflow.start_run(experiment_id=experiment_id,nested=True):
        gridsearch = GridSearchCV(best_model, param_grid, cv=5,  scoring='f1', n_jobs=-1)
        gridsearch.fit(X_train, y_train)
        best_model = gridsearch.best_estimator_


    f1_test_score = f1_score(y_test, best_model.predict(X_test))
    logger.info(f" Tuned {best_model_name} test F1-score: {f1_test_score:.4f}")
    if champion_dict.get('test_f1', 0.0) < f1_test_score:
        logger.info(f" New champion: {best_model_name} (F1: {f1_test_score:.4f}) > {champion_dict['test_f1']:.4f}")
        champion_dict["classifier"] = best_model_name
        champion_dict["test_f1"] = round(f1_test_score, 4)
        return best_model
    else:
        logger.info(f" Champion remains: {champion_dict['classifier']} (F1: {champion_dict['f1_test']:.4f})")
        return champion_model