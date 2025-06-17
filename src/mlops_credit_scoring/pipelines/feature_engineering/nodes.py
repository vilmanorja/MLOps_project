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

def extract_transactions_features(transactions: pd.DataFrame) -> pd.DataFrame:
    loans_reference_date = pd.to_datetime('2024-01-31')
    #loans_reference_date = pd.to_datetime(loans_reference_date)
    end_date = loans_reference_date - pd.DateOffset(months=1)
    start_date = end_date - pd.DateOffset(years=1) + pd.DateOffset(days=1)

    transactions["Date"] = pd.to_datetime(transactions["Date"])
    transactions = transactions[(transactions["Date"] >= start_date) & (transactions["Date"] <= end_date)].copy()

    transactions["CustomerIdCreditNew"] = transactions["CustomerIdCreditNew"].fillna(0).astype(int)
    transactions["CustomerIdDebitNew"] = transactions["CustomerIdDebitNew"].fillna(0).astype(int)
    transactions = transactions[~((transactions.CustomerIdCreditNew == 0) & (transactions.CustomerIdDebitNew == 0))]

    transactions['Month'] = transactions["Date"].dt.to_period('M')

    credited = transactions[['Month', 'CustomerIdCreditNew', 'AmountMZN']].groupby(['Month', 'CustomerIdCreditNew']).sum(numeric_only=True).reset_index().rename(columns={'AmountMZN': 'Monthly_Income', 'CustomerIdCreditNew': 'CustomerId'})
    debited = transactions[['Month', 'CustomerIdDebitNew', 'AmountMZN']].groupby(['Month', 'CustomerIdDebitNew']).sum(numeric_only=True).reset_index().rename(columns={'AmountMZN': 'Monthly_Expenses', 'CustomerIdDebitNew': 'CustomerId'})

    credited = credited[credited["CustomerId"] != 0]
    debited = debited[debited["CustomerId"] != 0]

    avg_income = credited.groupby("CustomerId").agg(Avg_Monthly_Income=('Monthly_Income','mean'), Income_Stability=('Monthly_Income','std')).reset_index()
    avg_expenses = debited.groupby("CustomerId").agg(Avg_Monthly_expenses=('Monthly_Expenses','mean'), Expenses_Stability=('Monthly_Expenses','std')).reset_index()

    summary = pd.merge(avg_income, avg_expenses, on="CustomerId", how="outer").fillna(0)
    return summary

def extract_transactions_features_batch(transactions: pd.DataFrame, reference_dates: list[str]) -> dict:
    result = {}
    for ref_date in reference_dates:
        summary = extract_transactions_features(transactions, ref_date)
        suffix = ref_date.replace("-", "")
        result[f"customer_transactional_summary_{suffix}"] = summary
    return result

def extract_funds_features(funds: pd.DataFrame) -> pd.DataFrame:
    loans_reference_date = pd.to_datetime('2024-01-31')
    end_date = loans_reference_date - pd.DateOffset(months=1)
    start_date = end_date - pd.DateOffset(years=1) + pd.DateOffset(days=1)

    funds["Date"] = pd.to_datetime(funds["Date"])
    funds = funds[(funds["Date"] >= start_date) & (funds["Date"] <= end_date)].copy()

    summary = funds.groupby("CustomerId").agg(
        Avg_Monthly_Funds=("FundsBalance", "mean"),
        Funds_Stability=("FundsBalance", "std")
    ).reset_index()

    return summary

def extract_previous_loans_features(loans_hist: pd.DataFrame) -> pd.DataFrame:
    loans_reference_date = pd.to_datetime('2024-01-31')
    end_date = loans_reference_date - pd.DateOffset(months=1)

    loans_hist["SnapshotDate"] = pd.to_datetime(loans_hist["SnapshotDate"])
    filtered = loans_hist[loans_hist["SnapshotDate"] <= end_date].copy()

    latest = filtered.sort_values(['ContractId','SnapshotDate'], ascending=[True, False]).drop_duplicates('ContractId', keep='first')

    prev_loans = latest.groupby("CustomerNewId").agg(
        Previous_Loan_Count=("ContractId", "count"),
        Previous_Loans_Avg_Amount=("CreditAmount", "mean"),
        Previous_Loans_Std=("CreditAmount", "std")
    ).reset_index().rename(columns={"CustomerNewId": "CustomerId"})

    defaults = filtered[filtered["HasDefault"] == 1]
    prev_defaults = defaults.groupby("CustomerNewId").agg(
        Previous_Loan_Defaults=("ContractId", "count")
    ).reset_index().rename(columns={"CustomerNewId": "CustomerId"})

    summary = pd.merge(prev_loans, prev_defaults, on="CustomerId", how="outer").fillna(0)
    return summary

def extract_active_loans_features(loans_hist: pd.DataFrame) -> pd.DataFrame:
    loans_reference_date = pd.to_datetime('2024-01-31')
    end_date = loans_reference_date - pd.DateOffset(months=1)

    loans_hist["SnapshotDate"] = pd.to_datetime(loans_hist["SnapshotDate"])
    active = loans_hist[loans_hist["SnapshotDate"] == end_date].copy()
    active["CreditEOMStartDate"] = pd.to_datetime(active["CreditEOMStartDate"])

    summary = active.groupby("CustomerNewId").agg(
        Active_Loans_Count=("ContractId", "count"),
        Active_Loan_Amount_Total=("Outstanding", "sum")
    ).reset_index().rename(columns={"CustomerNewId": "CustomerId"})

    return summary

def extract_loans_features(loans: pd.DataFrame) -> pd.DataFrame:
    loans["CreditEOMStartDate"] = pd.to_datetime(loans["CreditEOMStartDate"])
    loans["CreditEOMEndDate"] = pd.to_datetime(loans["CreditEOMEndDate"])

    filtered = loans[loans["CreditType"] != "Unarranged Overdraft"].copy()

    filtered["Duration_Months"] = (
        (filtered["CreditEOMEndDate"].dt.year - filtered["CreditEOMStartDate"].dt.year) * 12 +
        (filtered["CreditEOMEndDate"].dt.month - filtered["CreditEOMStartDate"].dt.month)
    )

    filtered = filtered.rename(columns={"CustomerNewId": "CustomerId"})
    return filtered[["CustomerId", "CreditType", "CreditAmount", "Duration_Months", "NumberOfInstallmentsToPay", "PaymentFrequency", "HasDefault"]]


def merge_features_with_target_loans(
    loans_to_predict: pd.DataFrame,
    transactional_summary: pd.DataFrame,
    funds_summary: pd.DataFrame,
    previous_loans_summary: pd.DataFrame,
    active_loans_summary: pd.DataFrame
) -> pd.DataFrame:

    df = loans_to_predict.copy()

    df = df.merge(transactional_summary, on="CustomerId", how="left")
    df = df.merge(funds_summary, on="CustomerId", how="left")
    df = df.merge(previous_loans_summary, on="CustomerId", how="left")
    df = df.merge(active_loans_summary, on="CustomerId", how="left")

    return df.fillna(0)
