import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import os
import pickle


def feature_engineering( customers: pd.DataFrame , loans: pd.DataFrame,  funds: pd.DataFrame, transactions: pd.DataFrame, loans_hist: pd.DataFrame):

    log = logging.getLogger(__name__)

    
    
    return full_data