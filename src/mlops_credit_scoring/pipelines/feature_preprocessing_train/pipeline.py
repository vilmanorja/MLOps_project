
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import feature_preprocessing_train

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=feature_preprocessing_train,
            inputs="X_train",
            outputs=["fitted_preprocessor", "X_train_processed"],
            name="feature_preprocessing_train_node"
        )
    ])