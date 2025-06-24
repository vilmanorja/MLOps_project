
import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np  
import pickle

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

logger = logging.getLogger(__name__)

def model_predict(X: pd.DataFrame,
                  model: pickle.Pickler, columns: pickle.Pickler) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Predict using the trained model.

    Args:
    --
        X (pd.DataFrame): Serving observations.
        model (pickle): Trained model.

    Returns:
    --
        scores (pd.DataFrame): Dataframe with new predictions.
    """

    # Predict
    
    y_pred = model.predict(X[columns])

    
    # Predict probability (for class 1)
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X[columns])[:, 1]
    else:
        raise AttributeError("Model does not support predict_proba().")

    # Add predictions to dataframe
    X['y_pred'] = y_pred
    X['y_pred_proba'] = y_pred_proba

    # Create dictionary with predictions
    describe_servings = X.describe().to_dict()

    logger.info('Service predictions created.')
    logger.info('#servings: %s', len(y_pred))
    return X, describe_servings,pd.DataFrame(y_pred, columns=["HasDefault"]), pd.DataFrame(y_pred_proba, columns=["HasDefault"])