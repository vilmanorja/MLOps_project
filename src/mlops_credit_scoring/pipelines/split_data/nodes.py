"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

import hopsworks

conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]

def split_data(
    data: pd.DataFrame, parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits data into features and target training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters.yml.
    Returns:
        Split data.
    """

    if parameters["load_from_feature_store"]:
        project = hopsworks.login( api_key_value=credentials["feature_store"]["FS_API_KEY"], project=credentials["feature_store"]["FS_PROJECT_NAME"])
        fs = project.get_feature_store(name='novaims_mlops_project')

        fg_target = fs.get_feature_group('target_project', version=1).read()
        fg_num = fs.get_feature_group('numerical_features_project', version=1).read()
        fg_cat = fs.get_feature_group('categorical_features_project', version=1).read()
        df = pd.merge(fg_target, fg_num, left_on='index', right_on='index')
        df = pd.merge(df, fg_cat, left_on='index', right_on='index')

        column_map = {
            'index': 'index',  # if still needed
            'has_default': 'HasDefault',
            'customer_id': 'CustomerId',
            'credit_amount': 'CreditAmount',
            'duration_months': 'Duration_Months',
            'number_of_installments_to_pay': 'NumberOfInstallmentsToPay',
            'run_date': 'run_date',
            'avg_monthly_income': 'Avg_Monthly_Income',
            'income_stability': 'Income_Stability',
            'avg_monthly_expenses': 'Avg_Monthly_expenses',
            'expenses_stability': 'Expenses_Stability',
            'avg_monthly_funds': 'Avg_Monthly_Funds',
            'funds_stability': 'Funds_Stability',
            'previous_loan_count': 'Previous_Loan_Count',
            'previous_loans_avg_amount': 'Previous_Loans_Avg_Amount',
            'previous_loans_std': 'Previous_Loans_Std',
            'previous_loan_defaults': 'Previous_Loan_Defaults',
            'active_loans_count': 'Active_Loans_Count',
            'active_loan_amount_total': 'Active_Loan_Amount_Total',
            'yr_net_monthly_in': 'YrNetMonthlyIn',
            'age': 'Age',
            'is_employed': 'Is_Employed',
            'is_married': 'Is_Married',
            'credit_type': 'CreditType',
            'payment_frequency': 'PaymentFrequency',
            'seg_group': 'SegGroup',
            'a_m_l_risk_rating': 'AMLRiskRating'
        }

        df = df.rename(columns=column_map)
    else:
        df = data

    assert [col for col in df.columns if df[col].isnull().any()] == []
    y = df[parameters["target_column"]]
    X = df.drop(columns=parameters["target_column"], axis=1)
    X = X.drop(columns="CustomerId", axis=1)
    #X = X.drop(columns="run_date", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=parameters["test_fraction"], random_state=parameters["random_state"])

    return X_train, X_test, y_train, y_test

    #return X_train, X_test, y_train, y_test, X_train.columns


