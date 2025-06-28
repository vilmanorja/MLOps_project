import pandas as pd
import numpy as np
import pytest

import os
import sys 

full_path = os.getcwd()
sys.path.append(os.path.abspath("src"))
# sys.path.append(full_path + '/Project/MLOps_project/src')

# sys.path.append('/Users/vilmanorja/Library/CloudStorage/OneDrive-AaltoUniversity/Yliopisto/Maisterikurssit/Exchange courses/MLOps/Project/MLOps_project/src')

from mlops_credit_scoring.pipelines.split_data.nodes import split_data

def test_split_data():
    """Test the split_data function.
    """
    print(full_path + "/tests/pipelines/sample/sample_features.csv")
    df = pd.read_csv(full_path + "/tests/pipelines/sample/sample_features.csv") 

    # Define the parameters
    parameters = {
        'target_column': 'HasDefault',
        'random_state': 42,
        'test_fraction': 0.2,
        'load_from_feature_store': False
    }

    # Call the split_data function
    X_train, X_test, y_train, y_test = split_data(df, parameters)

    # Assert the existence of the datasets
    assert X_train is not None
    assert X_test is not None
    assert y_train is not None
    assert y_test is not None

    # Assert the shapes of the resulting datasets
    assert X_train.shape == (80, 24)
    assert X_test.shape == (20, 24)
    assert y_train.shape == (80,)
    assert y_test.shape == (20,)

# print(f'Test cleaning')
# test_split_data()
# print(f'All passed')




