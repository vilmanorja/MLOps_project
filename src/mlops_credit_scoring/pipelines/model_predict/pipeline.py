"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_predict


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_predict,
                inputs=["X_test_processed","production_model","production_columns"],
                outputs=["features_with_predict", "predict_describe"],
                name="predict",
            ),
        ]
    )
