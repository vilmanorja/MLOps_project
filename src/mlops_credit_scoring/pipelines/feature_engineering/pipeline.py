
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import feature_engineering


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=feature_engineering,
                inputs=["customers_raw_data", "loans_raw_data", "funds_raw_data", "transactions_raw_data", "loans_hist_raw_data"],
                outputs="full_data",
                name="feature_engineering",
            ),
        ]
    )
