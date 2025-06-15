import logging
from typing import Any, Dict, Tuple

from great_expectations.core import ExpectationSuite, ExpectationConfiguration
import great_expectations as gx
import os

from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

import numpy as np
import pandas as pd
from datetime import datetime
import re


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

        for i in ['no_of_dependents', 'segment_id', 'industry_id', 'legal_doc_name1_id', 'yr_net_monthly_in']:
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": i, "type_": "float64"},
                )
            )
        # NewId
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "new_id"}
            )
        )
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_of_type",
                kwargs={"column": "new_id", "type_": "int64"}
            )
        )
        for i in ['segment_id', 'industry_id', 'legal_doc_name1_id', 'new_id']:
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={
                        "column": i,
                        "min_value": 1,
                        "strict_min": False,
                        "max_value": None  # No upper bound
                    }
                )
            )
        for i in ['yr_net_monthly_in', 'no_of_dependents']:
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={
                        "column": i,
                        "min_value": 0,
                        "strict_min": False,
                        "max_value": None  # No upper bound
                    }
                )
            )
        for i in ['yr_net_monthly_in', 'no_of_dependents', 'segment_id', 'industry_id', 'legal_doc_name1_id', 'new_id']:
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": i}
                )
            )

    if feature_group == 'categorical_features':
        for i in ['customer_status','employment_status','gender','marital_status', 'placebrth', 'cust_type', 
                  'nationality', 'ocupation_desc', 'residence_code','residence_status','residence_type',
                  'seg_group','title', 'town_country', 'cust_type_1', 'habliter', 'province', 'district', 
                  'legal_doc_name1_id_description', 'legal_iss_date', 'legal_iss_auth', 'a_m_l_risk_rating']:
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": i}
                )
            )

    return expectation_suite_bank


def get_validation_results(checkpoint_result):
    # validation_result is a dictionary containing one key-value pair
    validation_result_key, validation_result_data = next(iter(checkpoint_result["run_results"].items()))

    # Accessing the 'actions_results' from the validation_result_data
    validation_result_ = validation_result_data.get('validation_result', {})

    # Accessing the 'results' from the validation_result_data
    results = validation_result_["results"]
    meta = validation_result_["meta"]
    use_case = meta.get('expectation_suite_name')
    
    
    df_validation = pd.DataFrame({},columns=["Success","Expectation Type","Column","Column Pair","Max Value",\
                                       "Min Value","Element Count","Unexpected Count","Unexpected Percent","Value Set","Unexpected Value","Observed Value"])
    
    
    for result in results:
        # Process each result dictionary as needed
        success = result.get('success', '')
        expectation_type = result.get('expectation_config', {}).get('expectation_type', '')
        column = result.get('expectation_config', {}).get('kwargs', {}).get('column', '')
        column_A = result.get('expectation_config', {}).get('kwargs', {}).get('column_A', '')
        column_B = result.get('expectation_config', {}).get('kwargs', {}).get('column_B', '')
        value_set = result.get('expectation_config', {}).get('kwargs', {}).get('value_set', '')
        max_value = result.get('expectation_config', {}).get('kwargs', {}).get('max_value', '')
        min_value = result.get('expectation_config', {}).get('kwargs', {}).get('min_value', '')

        element_count = result.get('result', {}).get('element_count', '')
        unexpected_count = result.get('result', {}).get('unexpected_count', '')
        unexpected_percent = result.get('result', {}).get('unexpected_percent', '')
        observed_value = result.get('result', {}).get('observed_value', '')
        if type(observed_value) is list:
            #sometimes observed_vaue is not iterable
            unexpected_value = [item for item in observed_value if item not in value_set]
        else:
            unexpected_value=[]
        
        df_validation = pd.concat([df_validation, pd.DataFrame.from_dict( [{"Success" :success,"Expectation Type" :expectation_type,"Column" : column,"Column Pair" : (column_A,column_B),"Max Value" :max_value,\
                                           "Min Value" :min_value,"Element Count" :element_count,"Unexpected Count" :unexpected_count,"Unexpected Percent":unexpected_percent,\
                                                  "Value Set" : value_set,"Unexpected Value" :unexpected_value ,"Observed Value" :observed_value}])], ignore_index=True)
        
    return df_validation


def test_data(df):
    # context = gx.get_context(context_root_dir = "//..//..//gx")
    full_path = os.getcwd()
    context = gx.get_context(context_root_dir = full_path.partition('src')[0] + '/gx')

    datasource_name = "project_data_raw"
    try:
        datasource = context.sources.add_pandas(datasource_name)
        logger.info("Data Source created.")
    except:
        logger.info("Data Source already exists.")
        datasource = context.datasources[datasource_name]

    
    validation_expectation_suite_numerical = build_expectation_suite("numerical_expectations_raw", "numerical_features")
    validation_expectation_suite_categorical = build_expectation_suite("categorical_expectations_raw", "categorical_features")

    context.add_or_update_expectation_suite(expectation_suite=validation_expectation_suite_numerical)
    context.add_or_update_expectation_suite(expectation_suite=validation_expectation_suite_categorical)

    # add data
    logger.info(f"The dataset contains {len(df.columns)} columns.")

    df.columns = [re.sub(r'(?<!^)(?=[A-Z])', '_', col).lower() for col in df.columns]
    df.columns = df.columns.str.replace('.', '', regex=False)
    logger.info(f"{df.columns}")

    numerical_features = df.select_dtypes('number').columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    df_numeric = df[numerical_features].reset_index()
    df_categorical = df[categorical_features].reset_index()

    logger.info(f"Number of columns processed: {len(df_numeric.columns) + len(df_categorical.columns)} columns.")

    data_asset_name = "raw_data"
    try:
        data_asset = datasource.add_dataframe_asset(name=data_asset_name, dataframe=df)
    except:
        logger.info("The data asset alread exists. The required one will be loaded.")
        data_asset = datasource.get_asset(data_asset_name)

    batch_request = data_asset.build_batch_request(dataframe=df)

    checkpoint_num = gx.checkpoint.SimpleCheckpoint(
        name="checkpoint_num_raw",
        data_context=context,
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": "numerical_expectations_raw",
            },
        ],
    )
    checkpoint_cat = gx.checkpoint.SimpleCheckpoint(
        name="checkpoint_cat_raw",
        data_context=context,
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": "categorical_expectations_raw",
            },
        ],
    )
    checkpoint_result_num = checkpoint_num.run()
    checkpoint_result_cat = checkpoint_cat.run()

    df_validation_num = get_validation_results(checkpoint_result_num)
    df_validation_cat = get_validation_results(checkpoint_result_cat)
    df_validation = pd.concat([df_validation_num, df_validation_cat], ignore_index=True)
    #base on these results you can make an assert to stop your pipeline

    # pd_df_ge = gx.from_pandas(df)

    # assert pd_df_ge.expect_column_values_to_be_of_type("duration", "int64").success == True
    # assert pd_df_ge.expect_column_values_to_be_of_type("marital", "str").success == True
    #assert pd_df_ge.expect_table_column_count_to_equal(23).success == False
    
    log = logging.getLogger(__name__)
    log.info("Data passed on the unit data tests")
    log.info(f'All raw data tests passed: {df_validation[df_validation.Success == False].empty}')

    return df_validation




