
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import test_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= test_data,
                inputs=["customers_raw_data", "funds_raw_data", "transactions_raw_data", "loans_raw_data", "loans_hist_raw_data", "params:run_date"],
                outputs=['reporting_tests_raw', "customers_ingested", "funds_ingested", "transactions_ingested", "loans_ingested", "loans_hist_ingested"],
                name="raw_data_tests",
            ),
        ]
    )
