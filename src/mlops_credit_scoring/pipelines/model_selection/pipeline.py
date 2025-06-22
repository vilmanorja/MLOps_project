"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_selection


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_selection,
                inputs=["X_train_preprocessed","X_test_preprocessed","y_train_preprocessed","y_test_preprocessed",
                        "production_model_metrics",
                        "production_model",
                        "parameters", "best_columns"],
                outputs="champion_model",
                name="model_selection",
            ),
        ]
    )
