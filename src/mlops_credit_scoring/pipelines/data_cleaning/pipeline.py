"""
This is a boilerplate pipeline 'data_cleaning'
generated using Kedro 0.19.12
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import clean_customers, clean_Funds, clean_Loans_hist, clean_loans_partitioned,clean_Transactions

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=clean_Funds,
            inputs="funds_validated",
            outputs="funds_cleaned",
            name="clean_funds_node"
        ),
        node(
            func=clean_Transactions,
            inputs="transactions_validated",
            outputs="transactions_cleaned",
            name="clean_transactions_node"
        ),
        node(
            func=clean_Loans_hist,
            inputs="loans_hist_validated",
            outputs="loans_hist_cleaned",
            name="clean_loans_hist_node"
        ),
        node(
            func=clean_loans_partitioned,
            inputs="loans_validated",
            outputs="loans_cleaned",
            name="clean_loans_partitioned_node"
        ),
        node(
            func=clean_customers,
            inputs="customers_validated",
            outputs="customers_cleaned",
            name="clean_customers_node"
        ),
    ])
