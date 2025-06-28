from prefect.deployments import Deployment
from prefect.client.schemas.schedules import CronSchedule
from flows.kedro_prefect_flow import (
    full_pipeline,
    flow_full_processing,
    flow_model_selection,
    flow_prediction,
    flow_data_ingestion
)

#from kedro_prefect_flow import orchestrator
from prefect.infrastructure import Process

# Deployment 1: Data Ingestion - Daily at 10 AM Lisbon Time
Deployment.build_from_flow(
    flow=flow_data_ingestion,
    name="data-ingestion-daily",
    work_queue_name="default",
    schedule=CronSchedule(cron="0 10 * * *", timezone="Europe/Lisbon"), 
    tags=["data-ingestion"]
).apply()


# Deployment 2: Full Preprocessing - Monthly on the 1st at 6 AM Lisbon Time
Deployment.build_from_flow(
    flow=flow_full_processing,
    name="full-processing-morning",
    work_queue_name="default",
    schedule=CronSchedule(cron="0 6 1 * *", timezone="Europe/Lisbon"),  
    tags=["training-preprocess"]
).apply()

# Deployment 3: Model Selection - Monthly on the 1st at 9 AM Lisbon Time
Deployment.build_from_flow(
    flow=flow_model_selection,
    name="model-selection-monthly",
    work_queue_name="default",
    schedule=CronSchedule(cron="0 9 1 * *", timezone="Europe/Lisbon"), 
    tags=["model-selection"]
).apply()

# Deployment 4: Prediction - Monthly on the 1st at 12 PM Lisbon Time
Deployment.build_from_flow(
    flow=flow_prediction,
    name="prediction-monthly",
    work_queue_name="default",
    schedule=CronSchedule(cron="0 12 1 * *", timezone="Europe/Lisbon"),  
    tags=["prediction"]
).apply()

# Deployment 5: Full Pipeline Orchestration 
Deployment.build_from_flow(
    flow=full_pipeline,
    name="full-pipeline-manual",
    work_queue_name="default",
    infrastructure=Process(),  
    tags=["manual"]
).apply()