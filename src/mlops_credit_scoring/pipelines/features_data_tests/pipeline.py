
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import data_tests

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= data_tests,
                inputs=["customer_features", "parameters"],
                outputs= "reporting_tests",
                name="ingestion",
            ),

        ]
    )
