
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import  split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= split_data,
                inputs=["customer_features","parameters"],
                outputs= ["X_train","X_test","y_train","y_test"],
                name="split",
            ),
        ]
    )
