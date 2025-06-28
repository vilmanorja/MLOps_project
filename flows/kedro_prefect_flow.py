from prefect import flow, task, get_run_logger
from typing import Optional, Dict, List, Union
from src.mlops_credit_scoring.run_kedro_pipeline import run_pipeline
import subprocess
import sys
import os

# Fix for emoji/log encoding issues on Windows
sys.stdout.reconfigure(encoding="utf-8")

# Ensure src path is accessible
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

@task
def run_kedro_task(pipeline_name: str):
    logger = get_run_logger()
    logger.info(f"Running Kedro pipeline: `{pipeline_name}`")

    try:
        run_pipeline(pipeline_name)
        logger.info(f"Kedro pipeline `{pipeline_name}` finished successfully.")
    except Exception as e:
        logger.error(f"Pipeline `{pipeline_name}` failed: {str(e)}")
        raise

# @task
# def run_kedro_ingestion(run_dates: Optional[List[str]] = None):
#     logger = get_run_logger()

#     if run_dates:
#         for run_date in run_dates:
#             logger.info(f"Running ingestion pipeline with run_date = {run_date}")
#             run_pipeline("ingestion", parameters={"run_date": run_date})
#             logger.info(f"Ingestion for {run_date} finished.")
#     else:
#         logger.info("Running ingestion pipeline with default parameters from YAML")
#         run_pipeline("ingestion")

@task
def run_kedro_ingestion():
    logger = get_run_logger()
    logger.info("Running ingestion pipeline")
    run_pipeline("ingestion")
    logger.info("Ingestion pipeline finished.")

@flow
def pipeline_flow(pipeline_name: str):
    run_kedro_task(pipeline_name)

# @flow(name="Data Ingestion Flow", description="Runs data ingestion pipeline")
# def flow_data_ingestion(run_dates: Optional[List[str]] = None):
#     run_kedro_ingestion(run_dates)

@flow(name="Data Ingestion Flow", description="Runs data ingestion pipeline")
def flow_data_ingestion():
    run_kedro_ingestion()

@flow(name="Data Processing Flow", description="Runs full preprocessing pipeline")
def flow_full_processing():
    pipeline_flow("data_cleaning")
    pipeline_flow("feature_engineering") 
    pipeline_flow("features_data_tests")
    pipeline_flow("split_data")
    pipeline_flow("feature_preprocessing_train")

  
@flow(name="Model Training Flow")
def flow_model_train():
    pipeline_flow("feature_selection")
    pipeline_flow("model_selection")
    pipeline_flow("model_train")


@flow(name="Prediction Flow")
def flow_prediction():
    run_kedro_task("production_full_prediction_process")


@flow(name="Orchestration Flow")
def full_pipeline():
    """
    Full pipeline orchestration flow that runs all stages in sequence.
    This is intended for manual execution and can be triggered as needed.
    """
    flow_data_ingestion()
    flow_full_processing()
    flow_model_train()
    flow_prediction()

# def full_pipeline(run_dates: Optional[List[str]] = None):
    # flow_data_ingestion(run_dates)
    # flow_full_processing()
    # flow_model_train()
    # flow_prediction()
