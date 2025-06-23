"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""
import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder , LabelEncoder


from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]


logger = logging.getLogger(__name__)


"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Some function for cleaning customer final  features
def clean_customer_features_train(X_train: pd.DataFrame, y_train: pd.DataFrame) -> pd.DataFrame:
    X_train= X_train.drop(columns="run_date", axis=1)

    X_train_cleaned = X_train.drop_duplicates(keep='first')
    y_train_cleaned = y_train.loc[X_train_cleaned.index]

    #Outliers limit
    limits = {
    'CreditAmount': 6e8,
    'NumberOfInstallmentsToPay': 150,
    'Avg_Monthly_Income': 4e8,
    'Avg_Monthly_expenses':4e8,
    'Age': 100,
    'Expenses_Stability':1.5e8,
    'Avg_Monthly_Funds': 2.5e8,
    'Funds_Stability': 5e7,
    'Previous_Loan_Count': 60,
    'Previous_Loans_Avg_Amount':3e8,
    'Previous_Loans_Std': 1e8,
    'Active_Loans_Count':20,
    'Active_Loan_Amount_Total': 1.5e8,
    'YrNetMonthlyIn':1.5e7    
    }


    #Removing outliers
    for column, upper_limit in limits.items():
        X_train_cleaned = X_train_cleaned[X_train_cleaned[column] <= upper_limit]

    X_train_cleaned = X_train_cleaned[X_train_cleaned['Age'] > 0] 

    y_train_cleaned = y_train_cleaned.loc[X_train_cleaned.index]

    return X_train_cleaned, y_train_cleaned

def feature_preprocessing_train(X_train: pd.DataFrame, y_train: pd.DataFrame) -> tuple:

    X_train_cleaned, y_train_cleaned = clean_customer_features_train(X_train, y_train)

    numerical_cols = X_train_cleaned.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X_train_cleaned.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train_cleaned)

    try:
        feature_names = preprocessor.get_feature_names_out()
        X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names)
    except:
        X_train_processed = pd.DataFrame(X_train_processed)

    return preprocessor, X_train_processed, y_train_cleaned




