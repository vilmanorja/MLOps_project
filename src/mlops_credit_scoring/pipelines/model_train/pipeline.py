"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import model_train


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_train,
                inputs=["X_train_preprocessed","X_test_preprocessed","y_train_preprocessed","y_test",
                        "parameters","best_columns"],
                outputs=["production_model","production_columns" ,"production_model_metrics","output_plot"],
                name="train",
            ),
        ]
    )