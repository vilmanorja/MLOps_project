import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from datetime import datetime
import re

import great_expectations as gx
import os
from great_expectations.core import ExpectationSuite, ExpectationConfiguration
# from great_expectations.core.expectation_suite import ExpectationSuite
# from great_expectations.expectations.expectation_configuration import ExpectationConfiguration
#import great_expectations as gx

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
    
    expectation_suite = ExpectationSuite(
        expectation_suite_name=expectation_suite_name
    )
    
    #context = gx.get_context()
    #expectation_suite = context.add_expectation_suite("my_suite")
    #context.save_expectation_suite(expectation_suite)

    # target
    if feature_group == 'target':
        expectation_suite.add_expectation(
                    ExpectationConfiguration(
                        expectation_type="expect_column_values_to_be_of_type",
                        kwargs={"column": 'has_default', "type_": "int64"},
                    )
                )
        expectation_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_in_set",
                    kwargs={
                        "column": "has_default",
                        "value_set": [0, 1]  # or ["yes", "no"]
                    }
                )
            )
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": 'has_default'}
            )
        )
    # numerical features
    if feature_group == 'numerical_features':
        
        integer_features = ['customer_id', 'duration_months', 'number_of_installments_to_pay',
       'run_date', 'previous_loan_count', 'previous_loan_defaults', 'active_loans_count', 'age', 'is_employed', 'is_married']
        float_features = ['credit_amount', 'avg_monthly_income', 'income_stability',
       'avg_monthly_expenses', 'expenses_stability', 'avg_monthly_funds',
       'funds_stability', 'previous_loans_avg_amount', 'previous_loans_std',
       'active_loan_amount_total', 'yr_net_monthly_in']
        
        # int
        for i in integer_features:
                expectation_suite.add_expectation(
                    ExpectationConfiguration(
                        expectation_type="expect_column_values_to_be_of_type",
                        kwargs={"column": i, "type_": "int64"},
                    )
                )
        for i in integer_features:
            expectation_suite.add_expectation(
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
        # Id
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_not_be_null",
                kwargs={"column": "customer_id"}
            )
        )
        # float
        for i in float_features:
            expectation_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": i, "type_": "float64"}
                )
            )
        for i in float_features:
            expectation_suite.add_expectation(
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
        
        for i in ['is_employed', 'is_married']:
            expectation_suite.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_in_set",
                    kwargs={
                        "column": i,
                        "value_set": [0, 1]  # or ["yes", "no"]
                    }
                )
            )
        
    
    # categorical features
    if feature_group == 'categorical_features':

        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "seg_group", "value_set": ['Company', 'Personal']},
            )
        ) 
       
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "a_m_l_risk_rating", "value_set": ['Elevado', 'Medio', 'Baixo', '0']},
            )
        ) 
        expectation_suite.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "payment_frequency", "value_set": ['Monthly', 'Single', 'Semiannual', 'Quarterly']},
            )
        ) 
        expectation_suite.add_expectation(
                    ExpectationConfiguration(
                        expectation_type="expect_column_distinct_values_to_be_in_set",
                        kwargs={"column": "credit_type", "value_set": ['Credit Card', 'Unarranged Overdraft', 'Personal Credit',
                                                                        'Arranged Overdraft', 'Bill of Exchange Discount', 'Leasing',
                                                                        'Business Loan Account', 'Secured Current Account',
                                                                        'Mortgage Loan Account', 'Personal Loan Account']},
                    )
                ) 
    return expectation_suite

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

def data_tests(
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
        
    Returns: report
    """
    # get context
    full_path = os.getcwd()
    context = gx.get_context(context_root_dir = full_path.partition('src')[0] + '/gx')

    # add datasource
    datasource_name = "project_data_feature_engineered"
    try:
        datasource = context.sources.add_pandas(datasource_name)
        logger.info("Data Source created.")
    except:
        logger.info("Data Source already exists.")
        datasource = context.datasources[datasource_name]

    # df = pd.merge(customers, features, left_on='NewId', right_on='CustomerId', how='right').drop(columns=['NewId', 'index'])
    logger.info(f"The dataset contains {len(df.columns)} columns.")
    logger.info(df.columns)

    for c in ['CustomerId', 'Duration_Months', 'NumberOfInstallmentsToPay', 'HasDefault',
    'run_date', 'Previous_Loan_Count', 'Previous_Loan_Defaults', 'Active_Loans_Count',
    'Age', 'Is_Employed', 'Is_Married']:
        df[c] = df[c].fillna(-1)
        df[c] = df[c].astype('int')

    df.columns = [re.sub(r'(?<!^)(?=[A-Z])', '_', col).lower() for col in df.columns]
    df.columns = df.columns.str.replace('.', '', regex=False)
    df.columns = df.columns.str.replace('__', '_', regex=False)

    target = ['has_default']
    numerical_features = df.select_dtypes('number').drop(columns='has_default').columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
 
    for c in categorical_features:
        df[c] = df[c].apply(lambda x: x if pd.notnull(x) else None)

    validation_expectation_suite_target = build_expectation_suite("target_expectations","target")
    validation_expectation_suite_numerical = build_expectation_suite("numerical_expectations","numerical_features")
    validation_expectation_suite_categorical = build_expectation_suite("categorical_expectations","categorical_features")
    # validation_expectation_suite_datetime = build_expectation_suite("datetime_expectations","datetime_features")

    context.add_or_update_expectation_suite(expectation_suite=validation_expectation_suite_target)
    context.add_or_update_expectation_suite(expectation_suite=validation_expectation_suite_numerical)
    context.add_or_update_expectation_suite(expectation_suite=validation_expectation_suite_categorical)
    # context.add_or_update_expectation_suite(expectation_suite=validation_expectation_suite_datetime)

    numerical_feature_descriptions = [
        {
            "name": "index",
            "description": """
                            Index of the event. Key element to make joins.
                            """
        },
        {
            "name": "avg_monthly_funds",
            "description": """
                            Average funds per customer per month
                            """
        },
        {
            "name": "funds_stability",
            "description": """
                            Standard deviation of funds
                            """
        },
        {
            "name": "avg_monthly_income",
            "description": """
                            Average deposits in the customer account per month
                            """
        },
        {
            "name": "income_stability",
            "description": """
                            Standard deviation of income
                            """
        },
        {
            "name": "avg_monthly_expenses",
            "description": """
                            Average spending in the customer account per month
                            """
        },
        {
            "name": "expenses_stability",
            "description": """
                            Average spending per month
                            """
        },
        {
            "name": "previous_loan_count",
            "description": """
                            Number of previous loans the customer has taken
                            """
        },
        {
            "name": "previous_loans_avg_amount",
            "description": """
                            Credit card maximum limit
                            """
        },
        {
            "name": "previous_loans_std",
            "description": """
                            Standard deviation of past loan amounts
                            """
        },
        {
            "name": "previous_loan_defaults",
            "description": """
                            Number of times the customer defaulted on past loans
                            """
        },
        {
            "name": "active_loans_count",
            "description": """
                            Number of loans currently active
                            """
        },
        {
            "name": "active_loan_amount_total",
            "description": """
                            Total outstanding loan balance
                            """
        }
    ]
    categorical_feature_descriptions =[]
    target_description =[]

    df = df.reset_index()
    df_target = df[target + ['index']]
    df_numeric = df[numerical_features + ['index']]
    df_categorical = df[categorical_features + ['index']]
    # df_datetime = df[datetime_features].reset_index()

    logger.info(f"Number of columns processed: {len(df_numeric.columns) + len(df_categorical.columns) + len(df_target.columns)} columns.")
    logger.info(f"{categorical_features}")

    def build_asset_and_checkpoint(asset_name, df_group, checkpoint_name, expectation_suite_name):
        data_asset_name = asset_name
        try:
            data_asset = datasource.add_dataframe_asset(name=data_asset_name, dataframe=df_group)
        except:
            data_asset = datasource.get_asset(data_asset_name)

        batch_request = data_asset.build_batch_request(dataframe=df_group)

        checkpoint = gx.checkpoint.SimpleCheckpoint(
            name=checkpoint_name,
            data_context=context,
            validations=[
                {
                    "batch_request": batch_request,
                    "expectation_suite_name": expectation_suite_name,
                },
            ],
        )
        return checkpoint

    checkpoint_target = build_asset_and_checkpoint('target', df_target, 'checkpoint_target', 'target_expectations').run()
    checkpoint_num = build_asset_and_checkpoint('numerical_features', df_numeric, 'checkpoint_num', 'numerical_expectations').run()
    checkpoint_cat = build_asset_and_checkpoint('categorical_features', df_categorical, 'checkpoint_cat', 'categorical_expectations').run()
    # checkpoint_date = build_asset_and_checkpoint('date_features', df_datetime, 'checkpoint_date', 'datetime_expectations').run()

    df_val1 = get_validation_results(checkpoint_target)
    df_val2 = get_validation_results(checkpoint_num)
    df_val3 = get_validation_results(checkpoint_cat)
    # df_val4 = get_validation_results(checkpoint_date)
    
    df_validation = pd.concat([df_val1, df_val2, df_val3], ignore_index=True)
    logger.info(df_validation[df_validation.Success == False])
    logger.info("Data passed on the unit data tests")
    logger.info(f'All raw data tests passed: {df_validation[df_validation.Success == False].empty}')

    if parameters["to_feature_store"]:

        object_fs_numerical_features = to_feature_store(
            df_numeric,"numerical_features_project",
            1,"Numerical Features",
            numerical_feature_descriptions,
            validation_expectation_suite_numerical,
            credentials["feature_store"]
        )

        object_fs_categorical_features = to_feature_store(
            df_categorical,"categorical_features_project",
            1,"Categorical Features",
            categorical_feature_descriptions,
            validation_expectation_suite_categorical,
            credentials["feature_store"]
        )

        object_fs_target = to_feature_store(
            df_target,"target_project",
            1,"Target",
            target_description,
            validation_expectation_suite_target,
            credentials["feature_store"]
        )

    return df_validation

