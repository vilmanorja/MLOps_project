"""
This is a boilerplate pipeline 'data_cleaning'
generated using Kedro 0.19.12
"""
#imports
import pandas as pd

def clean_Funds(funds:pd.DataFrame) -> pd.DataFrame:
    """Converting 'Date' to datetime."""

    funds['Date']= pd.to_datetime(funds["Date"], errors="coerce")

    return funds

def clean_Transactions(transactions:pd.DataFrame) -> pd.DataFrame:
    """
    This function cleans the transactions DataFrame by:
    1. Converting the 'Date' column to datetime format.
    2. Filling NaN values in 'CustomerIdCreditNew' and 'CustomerIdDebitNew' with 0
       and converting them to integers.
    3. Dropping rows where both 'CustomerIdCreditNew' and 'CustomerIdDebitNew' are 0,
       ensuring that we only keep transactions that involve at least one customer"""

    # Converting 'Date' to datetime
    transactions['Date']= pd.to_datetime(transactions["Date"], errors="coerce")

    # Filling NaN values in 'CustomerIdCreditNew' and 'CustomerIdDebitNew' with 0
    # and converting them to integers
    transactions["CustomerIdCreditNew"] = transactions["CustomerIdCreditNew"].fillna(0).astype(int)
    transactions["CustomerIdDebitNew"] = transactions["CustomerIdDebitNew"].fillna(0).astype(int)

    # Dropping rows where both 'CustomerIdCreditNew' and 'CustomerIdDebitNew' are 0
    # This ensures that we only keep transactions that involve at least one customer
    transactions = transactions[~((transactions["CustomerIdCreditNew"] == 0) & (transactions["CustomerIdDebitNew"] == 0))]

    return transactions

def clean_Loans_hist(loans_hist:pd.DataFrame) -> pd.DataFrame:
    
    # Convert date columns
    date_cols = ["SnapshotDate", "CreditEOMStartDate", "CreditEOMEndDate"]
    for col in date_cols:
        loans_hist[col] = pd.to_datetime(loans_hist[col], errors="coerce")
    
    # Drop rows with NaN in 'CreditAmount' (same rows that have NaN in 'CreditEOMStartDate')
    loans_hist = loans_hist.dropna(subset=["CreditAmount"])

    # Fill NaN values in 'NumberOfInstallmentsToPay' with 0
    loans_hist["NumberOfInstallmentsToPay"] = loans_hist["NumberOfInstallmentsToPay"].fillna(0)
    
    # # Case 1: Contract ended → set installments = 0
    # ended_mask = (
    #     loans_hist["CreditEOMEndDate"].notna() &
    #     loans_hist["NumberOfInstallmentsToPay"].isna()
    # )
    # loans_hist.loc[ended_mask, "NumberOfInstallmentsToPay"] = 0

    # # Case 2: Active contract + PaymentFrequency is 'Single' → set installments = 0
    # single_active_mask = (
    #     loans_hist["CreditEOMEndDate"].isna() &
    #     (loans_hist["PaymentFrequency"] == "Single") &
    #     loans_hist["NumberOfInstallmentsToPay"].isna()
    # )
    # loans_hist.loc[single_active_mask, "NumberOfInstallmentsToPay"] = 0

    # # Case 3: All other missing values → fill with median
    # remaining_na_mask = loans_hist["NumberOfInstallmentsToPay"].isna()
    # median_value = loans_hist["NumberOfInstallmentsToPay"].median()
    # loans_hist.loc[remaining_na_mask, "NumberOfInstallmentsToPay"] = median_value

    return loans_hist

from typing import Dict
import pandas as pd

from typing import Dict, Callable
import pandas as pd

def clean_loans_partitioned(loans: Dict[str, Callable[[], pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    """
    Cleans a partitioned dataset of loan files. Each file is lazily loaded using a callable.

    Args:
        loans (Dict[str, Callable[[], pd.DataFrame]]): A dictionary where each key is a filename
            and each value is a callable that returns a DataFrame when invoked.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary with the same keys, containing the cleaned DataFrames.
    """
    cleaned = {}

    for file_name, load_dataframe in loans.items():
        try:
            df = load_dataframe()  # Load the actual DataFrame
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            continue

        if not isinstance(df, pd.DataFrame):
            print(f"{file_name} did not return a DataFrame. Got {type(df)} instead.")
            continue

        # Convert date columns to datetime
        date_cols = ["SnapshotDate", "CreditEOMStartDate", "CreditEOMEndDate"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Drop rows with missing values in 'CreditAmount'
        if "CreditAmount" in df.columns:
            df = df.dropna(subset=["CreditAmount"])

        # Fill missing values in 'NumberOfInstallmentsToPay' with 0
        if "NumberOfInstallmentsToPay" in df.columns:
            df["NumberOfInstallmentsToPay"] = df["NumberOfInstallmentsToPay"].fillna(0)

        # Save cleaned DataFrame
        cleaned[file_name] = df

    return cleaned


def clean_customers(customers: pd.DataFrame) -> pd.DataFrame:
    
    # Convert date columns to datetime
    date_cols = ["DateOfBirth", "BirthInCorpDate"]
    for col in date_cols:
        customers[col] = pd.to_datetime(customers[col], errors="coerce")
    
    # Fill NaN values in 'YrNetMonthlyIn' with 0
    customers["YrNetMonthlyIn"] = customers["YrNetMonthlyIn"].fillna(0)
    
    # Delete rows with NaN in 'SegGroup'
    customers = customers[~customers["SegGroup"].isna()].copy()
    
    return customers

   