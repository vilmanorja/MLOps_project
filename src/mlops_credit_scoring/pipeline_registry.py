# """Project pipelines."""
# from __future__ import annotations

# from kedro.framework.project import find_pipelines
# from kedro.pipeline import Pipeline


# def register_pipelines() -> dict[str, Pipeline]:
#     """Register the project's pipelines.

#     Returns:
#         A mapping from pipeline names to ``Pipeline`` objects.
#     """
#     pipelines = find_pipelines()
#     pipelines["__default__"] = sum(pipelines.values())
#     return pipelines




"""Project pipelines."""
from typing import Dict
from kedro.pipeline import Pipeline, pipeline

from mlops_credit_scoring.pipelines import (
    ingestion as ingestion,
    data_cleaning,
    # data_unit_tests as data_tests,
    feature_engineering as feature_engineering,
    features_data_tests as features_data_tests,
    feature_preprocessing_train,
    #split_train_pipeline as split_train,
    model_selection as model_selection_pipeline,
    model_train as model_train_pipeline,
    feature_selection as feature_selection_pipeline,
    split_data,
    feature_preprocessing_test,
    model_predict,
    data_drift,

)

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    ingestion_pipeline = ingestion.create_pipeline()
    features_data_tests_pipeline = features_data_tests.create_pipeline()
    # data_unit_tests_pipeline = data_tests.create_pipeline()
    data_cleaning_pipeline = data_cleaning.create_pipeline()
    feature_engineering_pipeline = feature_engineering.create_pipeline()
    split_data_pipeline = split_data.create_pipeline()
    preprocess_train_pipeline = feature_preprocessing_train.create_pipeline()
    preprocess_test_pipeline = feature_preprocessing_test.create_pipeline()
    #split_train_pipeline = split_train.create_pipeline()
    model_train = model_train_pipeline.create_pipeline()
    model_selection = model_selection_pipeline.create_pipeline()
    feature_selection = feature_selection_pipeline.create_pipeline()

    model_predict_pipeline = model_predict.create_pipeline()
    
    data_drift_pipeline = data_drift.create_pipeline()
    return {
        "__default__" : (ingestion_pipeline
                                           +data_cleaning_pipeline
                                           +feature_engineering_pipeline
                                           +features_data_tests_pipeline
                                           +split_data_pipeline 
                                           +preprocess_train_pipeline
                                           +preprocess_test_pipeline
                                           +feature_selection
                                           + model_selection
                                            + model_train
                                            +model_predict_pipeline
                                            +data_drift_pipeline),
        "ingestion": ingestion_pipeline,
        "features_data_tests": features_data_tests_pipeline,
        "data_cleaning": data_cleaning_pipeline,
        # "data_unit_tests": data_unit_tests_pipeline,
        "split_data": split_data_pipeline,
        "feature_engineering":feature_engineering_pipeline,
        "feature_preprocessing_train": preprocess_train_pipeline,
        #"split_train": split_train_pipeline,
        "model_selection": model_selection,
        "model_train": model_train,
        "feature_selection":feature_selection,
        "production_full_train_process" : (ingestion_pipeline
                                           +data_cleaning_pipeline
                                           +feature_engineering_pipeline
                                           # +features_data_tests_pipeline
                                           +split_data_pipeline 
                                           +preprocess_train_pipeline
                                           +preprocess_test_pipeline
                                           +feature_selection
                                             + model_train),
        "production_full_model_selection_process":(ingestion_pipeline
                                           +data_cleaning_pipeline
                                           +feature_engineering_pipeline
                                           #+features_data_tests_pipeline
                                           +split_data_pipeline 
                                           +preprocess_train_pipeline
                                           +feature_selection
                                             + model_selection),
        "feature_preprocessing_test": preprocess_test_pipeline,
        "model_predict" : model_predict_pipeline,
        "production_full_prediction_process" : preprocess_test_pipeline + model_predict_pipeline,
        "data_drift": data_drift_pipeline,
  
    }