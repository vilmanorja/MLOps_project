
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (  # assuming your functions are in nodes.py
    extract_transactions_features,
    extract_funds_features,
    extract_previous_loans_features,
    extract_active_loans_features,
    extract_loans_features,
    merge_features_with_target_loans,
    extract_transactions_features_batch
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=extract_transactions_features,
            inputs="transactions_raw_data",
            outputs="transactional_summary",
            name="transaction_features_node",
        ),
        node(
            func=extract_funds_features,
            inputs="funds_raw_data",
            outputs="funds_summary",
            name="funds_features_node",
        ),
        node(
            func=extract_previous_loans_features,
            inputs="loans_hist_raw_data",
            outputs="previous_loans_summary",
            name="previous_loans_features_node",
        ),
        node(
            func=extract_active_loans_features,
            inputs="loans_hist_raw_data",
            outputs="active_loans_summary",
            name="active_loans_features_node",
        ),
        node(
            func=extract_loans_features,
            inputs="loans_raw_data",
            outputs="loans_to_predict",
            name="loans_features_node",
        ),
        # node(
        #     func=extract_transactions_features_batch,
        #     inputs=["transactions_raw_data", "params:reference_dates"],
        #     outputs="transactional_summaries",
        #     name="transaction_features_batch_node"
        # ),
        node(
            func=merge_features_with_target_loans,
            inputs=[
                "loans_to_predict",
                "transactional_summary",
                "funds_summary",
                "previous_loans_summary",
                "active_loans_summary"
            ],
            outputs="behavior_features",
            # outputs="model_input_table"
            name="merge_features_node",
        ),
        ])