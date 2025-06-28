"""
This module contains an example test.

Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py. They are simply functions
named ``test_*`` which test a unit of logic.

To run the tests, run ``kedro test`` from the project root directory.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import os
import yaml
import sys 

full_path = os.getcwd()
sys.path.append(os.path.abspath("src"))

from mlops_credit_scoring.pipelines.data_cleaning.nodes import clean_Transactions
from mlops_credit_scoring.pipelines.feature_engineering.nodes import extract_transactions_features_batch

def test_clean_transactions():
    df = pd.read_csv(full_path + "/tests/pipelines/sample/sample_transactions.csv") 
    df_transformed  = clean_Transactions(df)
    assert [col for col in ["CustomerIdCreditNew", "CustomerIdDebitNew"] if df_transformed[col].isnull().any()] == []
    assert(isinstance(df_transformed, pd.DataFrame))

def test_feature_engineering_transactions():
    run_date = ['20240131']
    df = pd.read_csv(full_path + "/tests/pipelines/sample/sample_transactions_cleaned.csv") 
    df_transformed = extract_transactions_features_batch(df, run_date)
    assert(isinstance(df_transformed, dict))
    assert(set(['CustomerId','Avg_Monthly_Income','Income_Stability','Avg_Monthly_expenses','Expenses_Stability','run_date']).issubset(df_transformed['customer_transactional_summary_20240131'].columns))

# print(f'Test cleaning')
# test_clean_transactions()
# print(f'Test engineering')
# test_feature_engineering_transactions()
# print(f'All passed')