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


# Some function for cleaning customer final  features
def clean_customer_features(X_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
    X_test= X_test.drop(columns="run_date", axis=1)
    X_test_cleaned = X_test.drop_duplicates(keep='first')
    y_test_cleaned = y_test.loc[X_test_cleaned.index]

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

    
    for column, upper_limit in limits.items():
        X_test_cleaned = X_test_cleaned[X_test_cleaned[column] <= upper_limit]

    X_test_cleaned = X_test_cleaned[X_test_cleaned['Age'] > 0] 

    y_test_cleaned = y_test_cleaned.loc[X_test_cleaned.index]

    return X_test_cleaned, y_test_cleaned

def feature_preprocessing_test( X_test: pd.DataFrame,y_test: pd.DataFrame, preprocessor) -> pd.DataFrame:
    
    log = logging.getLogger(__name__)

    X_test_cleaned, y_test_cleaned = clean_customer_features(X_test, y_test)

    X_test_processed = preprocessor.transform(X_test_cleaned)

    try:
        feature_names = preprocessor.get_feature_names_out()
        X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names)
    except:
        X_test_processed = pd.DataFrame(X_test_processed)
    log.info(f"The final dataframe has {len(X_test_processed.columns)} columns.")

    return X_test_processed, y_test_cleaned


