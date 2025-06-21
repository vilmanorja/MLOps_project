
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (  # assuming your functions are in nodes.py
    extract_funds_features_batch,
    extract_previous_loans_features_batch,
    extract_active_loans_features_batch,
    extract_loans_features_batch,
    merge_features_with_target_loans_batch,
    extract_transactions_features_batch,
    extract_customer_demographics_features_batch
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=extract_funds_features_batch,
            inputs=["funds_cleaned","params:run_date"],
            outputs="funds_summaries",
            name="funds_features_node",
        ),
        node(
            func=extract_previous_loans_features_batch,
            inputs=["loans_hist_cleaned", "params:run_date"],
            outputs="prev_loans_summaries",
            name="previous_loans_features_node",
        ),
        node(
            func=extract_active_loans_features_batch,
            inputs=["loans_hist_cleaned", "params:run_date"],
            outputs="active_loans_summaries",
            name="active_loans_features_node",
        ),
        node(
            func=extract_loans_features_batch,
            inputs=["loans_cleaned", "params:run_date"],
            outputs="loans_to_predict",
            name="loans_features_batch_node"
        )
        ,node(
            func=extract_transactions_features_batch,
            inputs=["transactions_cleaned", "params:run_date"],
            outputs="transactional_summaries",
            name="transaction_features_batch_node"
        )
        ,
        node(
            func=extract_customer_demographics_features_batch,
            inputs=["customers_cleaned", "loans_to_predict", "params:run_date"],
            outputs="customer_demographics_features",
            name="extract_customer_demographics_features_batch_node"
        )
        ,node(
            func=merge_features_with_target_loans_batch,
            inputs=[
                "loans_to_predict",
                "transactional_summaries",
                "funds_summaries",
                "prev_loans_summaries",
                "active_loans_summaries",
                "customer_demographics_features",
                "params:run_date"
            ],
            outputs="customer_features",
            # outputs="model_input_table"
            name="merge_features_node",
        )
        ])