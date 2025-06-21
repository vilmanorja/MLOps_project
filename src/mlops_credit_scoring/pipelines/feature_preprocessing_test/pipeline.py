
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import feature_preprocessing_test


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=feature_preprocessing_test,
            inputs=["X_test", "fitted_preprocessor"],
            outputs="X_test_processed",
            name="feature_preprocessing_infer_node"
        )
    ])