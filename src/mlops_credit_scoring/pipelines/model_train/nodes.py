
import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np
import pickle
import yaml
import os
import warnings
warnings.filterwarnings("ignore", category=Warning)
import mlflow
from sklearn.metrics import accuracy_score, f1_score,recall_score
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def model_train(X_train: pd.DataFrame, 
                X_test: pd.DataFrame, 
                y_train: pd.DataFrame, 
                y_test: pd.DataFrame,
                parameters: Dict[str, Any], best_columns):
    """Trains a model on the given data and saves it to the given model path.

    Args:
    --
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.
        y_train (pd.DataFrame): Training target.
        y_test (pd.DataFrame): Test target.

    Returns:
    --
        model (pickle): Trained models.
        scores (json): Trained model metrics.
    """

    # enable autologging
    with open('conf/local/mlflow.yml') as f:
        experiment_name = yaml.load(f, Loader=yaml.loader.SafeLoader)['tracking']['experiment']['name']
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    logger.info('Starting first step of model selection : Comparing between modes types')
    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)

  # Load champion model or fallback to baseline
    try:
        with open("data/06_models/champion_model.pkl", "rb") as f:
            classifier = pickle.load(f)
        logger.info("Loaded existing champion model.")
    except Exception:
        classifier = RandomForestClassifier(**parameters["baseline_model_params"])
        logger.info("No champion found. Using baseline RandomForestClassifier.")

    
    if parameters["use_feature_selection"]:
        logger.info(f"Using feature selection in model train...")
        X_train = X_train[best_columns]
        X_test = X_test[best_columns]
    y_train = np.ravel(y_train)
    
    with mlflow.start_run(experiment_id=experiment_id, nested=True):

        model = classifier.fit(X_train, y_train)
        # making predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        # evaluating model
        acc_train = accuracy_score(y_train, y_train_pred)
        acc_test = accuracy_score(y_test, y_test_pred)
        f1_train = f1_score(y_train, y_train_pred)
        f1_test = f1_score(y_test, y_test_pred)
        recall_train = recall_score(y_train, y_train_pred)
        recall_test = recall_score(y_test, y_test_pred)
        # saving results in dict
        results_dict = {
            "classifier": classifier.__class__.__name__,
            "train_accuracy": round(acc_train, 4),
            "test_accuracy": round(acc_test, 4),
            "train_f1": round(f1_train, 4),
            "test_f1": round(f1_test, 4),
            "train_recall": round(recall_train, 4),
            "test_recall": round(recall_test, 4),
        }

        # logging in mlflow
        run_id = mlflow.last_active_run().info.run_id
        logger.info(f"Logged train model in run {run_id}")
        logger.info(f"Accuracy is {acc_test}")
        logger.info(f"F1-Score is {f1_test}")
        logger.info(f"Recall is {recall_test}")
        
        #logging the model and registering the model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="loan_default_model"
        )
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri, "loan_default_model")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_train)
        print("SHAP values shape:", np.array(shap_values).shape)
        shap.initjs()
        # calculate shap values. This is what we will plot.
        # shap_values[:,:,1] -> since it is a classification problem, I will use SHAP for explaining the outcome of class 1.
        # you can do the same for the class 0 just by using shap_values[:,:,0]
        
        shap.summary_plot(shap_values, X_train, feature_names=X_train.columns, show=False)
        #shap.summary_plot(shap_values[:,:,1], X_train,feature_names=X_train.columns, show=False)
    
    except Exception as e:
        logger.warning(f" SHAP failed: {e}")
        plt.figure()
        plt.text(0.5, 0.5, "SHAP unavailable", ha="center")
   
    return model, X_train.columns , results_dict,plt