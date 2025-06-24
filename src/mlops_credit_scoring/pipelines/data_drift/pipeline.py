"""
This is a boilerplate pipeline 'data_drift'
generated using Kedro 0.19.13
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import (
    combine_prediction_outputs,
    detect_data_drift,
    detect_multivariate_drift,
    estimate_performance
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=combine_prediction_outputs,
            inputs=["X_test", "y_preds", "y_pred_proba", "y_test"],
            outputs="customers_pred_test",
            name="combine_predictions_node"
        ),
        node(
            func=detect_data_drift,
            inputs=["customers_pred_test", "params:reference_split_date"],
            outputs="drift_univariate_results",
            name="univariate_drift_node"
        ),
        node(
            func=detect_multivariate_drift,
            inputs=["customers_pred_test", "params:reference_split_date"],
            outputs="drift_multivariate_results",
            name="multivariate_drift_node"
        ),
        node(
            func=estimate_performance,
            inputs=["customers_pred_test", "params:reference_split_date"],
            outputs="estimated_performance_results",
            name="performance_estimation_node"
        )
    ])