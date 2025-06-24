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

    #Removing outliers
    Q1_age = X_train_cleaned['Age'].quantile(0.25)
    Q3_age = X_train_cleaned['Age'].quantile(0.75)
    IQR_age = Q3_age - Q1_age
    lower_limit_age = Q1_age - 1.5 * IQR_age
    upper_limit_age = Q3_age + 1.5 * IQR_age
    X_train_cleaned = X_train_cleaned[(X_train_cleaned['Age'] >= lower_limit_age) & (X_train_cleaned['Age'] <= upper_limit_age)]

    columns_to_filter = [
        'CreditAmount', 'NumberOfInstallmentsToPay', 'Avg_Monthly_Income', 
        'Avg_Monthly_expenses', 'Expenses_Stability', 'Avg_Monthly_Funds',
        'Funds_Stability', 'Previous_Loan_Count', 'Previous_Loans_Avg_Amount', 
        'Previous_Loans_Std', 'Active_Loans_Count', 'Active_Loan_Amount_Total', 
        'YrNetMonthlyIn'
    ]

    # Remove outliers based on the 99th percentile for specified columns
    for column in columns_to_filter:
        upper_percentile = X_train_cleaned[column].quantile(0.99)  # 99% superior
        X_train_cleaned = X_train_cleaned[X_train_cleaned[column] <= upper_percentile]
 

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




