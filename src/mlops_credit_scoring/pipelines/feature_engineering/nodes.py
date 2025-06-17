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

def extract_transactions_features(transactions: pd.DataFrame, run_date:str) -> pd.DataFrame:
    loans_reference_date = pd.to_datetime(run_date, format="%Y%m%d")
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

def extract_transactions_features_batch(transactions: pd.DataFrame, run_date:list[str]) -> dict:
    result = {}
    for ref_date in run_date:
        summary = extract_transactions_features(transactions, ref_date)
        summary["run_date"]=ref_date
        suffix = ref_date.replace("-", "")
        result[f"customer_transactional_summary_{suffix}"] = summary

    return  result

def extract_funds_features(funds: pd.DataFrame, run_date:str) -> pd.DataFrame:
    #loans_reference_date = pd.to_datetime('2024-01-31')
    loans_reference_date = pd.to_datetime(run_date, format="%Y%m%d")
    end_date = loans_reference_date - pd.DateOffset(months=1)
    start_date = end_date - pd.DateOffset(years=1) + pd.DateOffset(days=1)

    funds["Date"] = pd.to_datetime(funds["Date"])
    funds = funds[(funds["Date"] >= start_date) & (funds["Date"] <= end_date)].copy()

    summary = funds.groupby("CustomerId").agg(
        Avg_Monthly_Funds=("FundsBalance", "mean"),
        Funds_Stability=("FundsBalance", "std")
    ).reset_index()

    return summary

def extract_funds_features_batch(funds: pd.DataFrame, run_date: list[str]) -> dict:
    result = {}
    for ref_date in run_date:
        summary = extract_funds_features(funds, ref_date)
        summary["run_date"] = ref_date
        result[f"customer_funds_summary_{ref_date}"] = summary
    return result

def extract_previous_loans_features(loans_hist: pd.DataFrame, run_date :str) -> pd.DataFrame:
    loans_reference_date = pd.to_datetime(run_date, format="%Y%m%d")
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

def extract_previous_loans_features_batch(loans_hist: pd.DataFrame, run_date: list[str]) -> dict:
    result = {}
    for ref_date in run_date:
        summary = extract_previous_loans_features(loans_hist, ref_date)
        summary["run_date"] = ref_date
        result[f"customer_prev_loans_summary_{ref_date}"] = summary
    return result

def extract_active_loans_features(loans_hist: pd.DataFrame, run_date:str) -> pd.DataFrame:
    loans_reference_date = pd.to_datetime(run_date, format="%Y%m%d")
    end_date = loans_reference_date - pd.DateOffset(months=1)
    loans_hist["SnapshotDate"] = pd.to_datetime(loans_hist["SnapshotDate"])
    active = loans_hist[loans_hist["SnapshotDate"] == end_date].copy()
    active["CreditEOMStartDate"] = pd.to_datetime(active["CreditEOMStartDate"])

    summary = active.groupby("CustomerNewId").agg(
        Active_Loans_Count=("ContractId", "count"),
        Active_Loan_Amount_Total=("Outstanding", "sum")
    ).reset_index().rename(columns={"CustomerNewId": "CustomerId"})

    return summary

def extract_active_loans_features_batch(loans_hist: pd.DataFrame, run_date: list[str]) -> dict:
    result = {}
    for ref_date in run_date:
        summary = extract_active_loans_features(loans_hist, ref_date)
        summary["run_date"] = ref_date
        result[f"customer_active_loans_summary_{ref_date}"] = summary
    return result

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

def extract_loans_features_batch(loans_files: dict[str, object], run_date: list[str]) -> dict:
    results = {}

    for date in run_date:
        key = f"Loans_{date}"  # No ".csv" suffix here
        dataset = loans_files.get(key)
        if dataset is None:
            print(f"{key} → SKIPPED")
            continue
        loans_df = dataset() # Must call .load()
        summary = extract_loans_features(loans_df)
        summary["run_date"] = date
        results[f"customer_loans_to_predict_{date}"] = summary

    return results


def merge_features_with_target_loans(
    loans_to_predict: pd.DataFrame,
    transactional_summary: pd.DataFrame,
    funds_summary: pd.DataFrame,
    previous_loans_summary: pd.DataFrame,
    active_loans_summary: pd.DataFrame
) -> pd.DataFrame:
    df = loans_to_predict.copy()

    df = df.merge(transactional_summary, on=["CustomerId", "run_date"], how="left")
    df = df.merge(funds_summary, on=["CustomerId", "run_date"], how="left")
    df = df.merge(previous_loans_summary, on=["CustomerId", "run_date"], how="left")
    df = df.merge(active_loans_summary, on=["CustomerId", "run_date"], how="left")

    return df.fillna(0)

def merge_features_with_target_loans_batch(
    loans_to_predict: dict[str, pd.DataFrame],
    transactional_summaries: dict[str, pd.DataFrame],
    funds_summaries: dict[str, pd.DataFrame],
    prev_loans_summaries: dict[str, pd.DataFrame],
    active_loans_summaries: dict[str, pd.DataFrame],
    run_date: list[str]
) -> pd.DataFrame:
    
    results = []

    for date in run_date:
        loans_df = loans_to_predict.get(f"customer_loans_to_predict_{date}")()
        tx_df = transactional_summaries.get(f"customer_transactional_summary_{date}")()
        funds_df = funds_summaries.get(f"customer_funds_summary_{date}")()
        prev_df = prev_loans_summaries.get(f"customer_prev_loans_summary_{date}")()
        active_df = active_loans_summaries.get(f"customer_active_loans_summary_{date}")()

        if loans_df is None:
            continue
        df_merged = merge_features_with_target_loans(
            loans_df,
            tx_df if tx_df is not None else pd.DataFrame(columns=["CustomerId"]),
            funds_df if funds_df is not None else pd.DataFrame(columns=["CustomerId"]),
            prev_df if prev_df is not None else pd.DataFrame(columns=["CustomerId"]),
            active_df if active_df is not None else pd.DataFrame(columns=["CustomerId"])
        )

        df_merged["run_date"] = date
        results.append(df_merged)

    return pd.concat(results, ignore_index=True)
