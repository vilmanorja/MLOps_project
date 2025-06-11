import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from datetime import datetime
import re

from great_expectations.core import ExpectationSuite, ExpectationConfiguration


from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]


logger = logging.getLogger(__name__)

def build_expectation_suite(expectation_suite_name: str, feature_group: str) -> ExpectationSuite:
    """
    Builder used to retrieve an instance of the validation expectation suite.
    
    Args:
        expectation_suite_name (str): A dictionary with the feature group name and the respective version.
        feature_group (str): Feature group used to construct the expectations.
             
    Returns:
        ExpectationSuite: A dictionary containing all the expectations for this particular feature group.
    """
    
    expectation_suite_bank = ExpectationSuite(
        expectation_suite_name=expectation_suite_name
    )
    

    # numerical features
    if feature_group == 'numerical_features':
        for i in ['no_of_dependents', 'segment_id', 'industry_id', 'legal_doc_name1_id', 'new_id']:
                expectation_suite_bank.add_expectation(
                    ExpectationConfiguration(
                        expectation_type="expect_column_values_to_be_of_type",
                        kwargs={"column": i, "type_": "int64"},
                    )
                )
        # NewId
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "new_id"}
            )
        )
        # YrNetMonthlyIn
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_of_type",
                kwargs={"column": "yr_net_monthly_in", "type_": "float64"}
            )
        )
        for i in ['yr_net_monthly_in', 'no_of_dependents', 'segment_id', 'industry_id', 'legal_doc_name1_id', 'new_id']:
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={
                        "column": i,
                        "min_value": -1,
                        "strict_min": False,
                        "max_value": None  # No upper bound
                    }
                )
            )
    # datetime features
    if feature_group == 'datetime_features':
        for i in ['customer_since', 'date_of_birth', 'birth_in_corp_date', 'legal_iss_date']:
            # expectation_suite_bank.add_expectation(
            #     ExpectationConfiguration(
            #         expectation_type="expect_column_values_to_be_of_type",
            #         kwargs={"column": i, 'type_':"datetime64[ns]"},
            #     )
            # ) 
            expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_match_strftime_format",
                kwargs={"column": i, 'strftime_format': '%Y-%m-%d'}
            )
        )
        # CustomerSince
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "customer_since"}
            )
        )
        # # legal_exp_date
        # expectation_suite_bank.add_expectation(
        #     ExpectationConfiguration(
        #         expectation_type="expect_column_values_to_match_strftime_format",
        #         kwargs={"column": 'legal_exp_date', 'strftime_format': '%Y-%m-%d'}
        #     )
        # )
        
    return expectation_suite_bank


import hopsworks

def to_feature_store(
    data: pd.DataFrame,
    group_name: str,
    feature_group_version: int,
    description: str,
    group_description: dict,
    validation_expectation_suite: ExpectationSuite,
    credentials_input: dict
):
    """
    This function takes in a pandas DataFrame and a validation expectation suite,
    performs validation on the data using the suite, and then saves the data to a
    feature store in the feature store.

    Args:
        data (pd.DataFrame): Dataframe with the data to be stored
        group_name (str): Name of the feature group.
        feature_group_version (int): Version of the feature group.
        description (str): Description for the feature group.
        group_description (dict): Description of each feature of the feature group. 
        validation_expectation_suite (ExpectationSuite): group of expectations to check data.
        SETTINGS (dict): Dictionary with the settings definitions to connect to the project.
        
    Returns:
        A dictionary with the feature view version, feature view name and training dataset feature version.
    
    
    """
    # Connect to feature store.
    project = hopsworks.login(
        api_key_value=credentials_input["FS_API_KEY"], project=credentials_input["FS_PROJECT_NAME"]
    )
    feature_store = project.get_feature_store()

    # Create feature group.
    object_feature_group = feature_store.get_or_create_feature_group(
        name=group_name,
        version=feature_group_version,
        description= description,
        primary_key=["index"],
        #event_time="datetime",
        online_enabled=False,
        expectation_suite=validation_expectation_suite,
    )
    # Upload data.
    object_feature_group.insert(
        features=data,
        overwrite=False,
        write_options={
            "wait_for_job": True,
        },
    )

    # Add feature descriptions.

    for description in group_description:
        object_feature_group.update_feature_description(
            description["name"], description["description"]
        )

    # Update statistics.
    object_feature_group.statistics_config = {
        "enabled": True,
        "histograms": True,
        "correlations": True,
    }
    object_feature_group.update_statistics_config()
    object_feature_group.compute_statistics()

    return object_feature_group


def ingestion(
    df: pd.DataFrame,
    parameters: Dict[str, Any]):

    """
    This function takes in a pandas DataFrame and a validation expectation suite,
    performs validation on the data using the suite, and then saves the data to a
    feature store in the feature store.

    Args:
        data (pd.DataFrame): Dataframe with the data to be stored
        group_name (str): Name of the feature group.
        feature_group_version (int): Version of the feature group.
        description (str): Description for the feature group.
        group_description (dict): Description of each feature of the feature group. 
        validation_expectation_suite (ExpectationSuite): group of expectations to check data.
        SETTINGS (dict): Dictionary with the settings definitions to connect to the project.
        
    Returns:
       
    
    
    """

    logger.info(f"The dataset contains {len(df.columns)} columns.")
    df.columns = [re.sub(r'(?<!^)(?=[A-Z])', '_', col).lower() for col in df.columns]
    df.columns = df.columns.str.replace('.', '', regex=False)
    logger.info(f"{df.columns}")

    for c in ['customer_since', 'date_of_birth', 'birth_in_corp_date']:
        df[c] = pd.to_datetime(df[c], format = "%Y-%m-%d", errors="coerce")
        df[c] = df[c].apply(
            lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else None
        )

    def safe_parse(val):
        try:
            if pd.isna(val):
                return None
            val_str = str(int(val))  # Convert float like 20250101.0 to '20250101'
            return datetime.strptime(val_str, "%Y%m%d").strftime('%Y-%m-%d')
        except ValueError:
            return None  # Or return None

    for c in ['legal_iss_date', 'legal_exp_date']:
        df[c] = df[c].str.split('ý').str[-1]
        df[c] = df[c].apply(safe_parse)

    # df['legal_exp_date'] = df['legal_exp_date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else None)

    for c in ['no_of_dependents', 'segment_id', 'industry_id', 'legal_doc_name1_id']:
        # Convert to Int64 (nullable int)
        df[c] = df[c].fillna(-1)
        df[c] = df[c].astype('int')

    numerical_features = df.select_dtypes('number').columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    datetime_features = ['customer_since', 'date_of_birth', 'birth_in_corp_date', 'legal_iss_date', 'legal_exp_date'] #df.select_dtypes(include=['datetime']).columns.tolist()
    
    for c in categorical_features:
        df[c] = df[c].apply(lambda x: x if pd.notnull(x) else None)

    validation_expectation_suite_numerical = build_expectation_suite("numerical_expectations","numerical_features")
    validation_expectation_suite_datetime = build_expectation_suite("datetime_expectations","datetime_features")

    numerical_feature_descriptions =[]
    categorical_feature_descriptions =[]
    datetime_feature_descriptions =[]
    
    df_numeric = df[numerical_features].reset_index()
    df_categorical = df[categorical_features].reset_index()
    df_datetime = df[datetime_features].reset_index()

    logger.info(f"Number of columns processed: {len(df_numeric.columns) + len(df_categorical.columns) + len(df_datetime.columns)} columns.")
    logger.info(f"{df_datetime.dtypes}")


    if parameters["to_feature_store"]:

        # object_fs_numerical_features = to_feature_store(
        #     df_numeric,"numerical_features_project",
        #     1,"Numerical Features",
        #     numerical_feature_descriptions,
        #     validation_expectation_suite_numerical,
        #     credentials["feature_store"]
        # )

        # object_fs_categorical_features = to_feature_store(
        #     df_categorical,"categorical_features_project",
        #     1,"Categorical Features",
        #     categorical_feature_descriptions,
        #     validation_expectation_suite_categorical,
        #     credentials["feature_store"]
        # )

        object_fs_datetime_features = to_feature_store(
            df_datetime,"datetime_features_project",
            1,"Target Features",
            datetime_feature_descriptions,
            validation_expectation_suite_datetime,
            credentials["feature_store"]
        )

    return df

