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

    # Customers features
    if feature_group == 'customers_features':

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

        for i in ['customer_status','employment_status','gender','marital_status', 'placebrth', 'cust_type', 
                  'nationality', 'ocupation_desc', 'residence_code','residence_status','residence_type',
                  'seg_group','title', 'town_country', 'cust_type1', 'habliter', 'province', 'district', 
                  'legal_doc_name1_id_description', 'legal_iss_date', 'legal_iss_auth', 'a_m_l_risk_rating']:
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": i}
                )
            )

    if feature_group == 'loans_features':

        for i in ['customer_new_id', 'contract_id', 'has_default']:
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": i, "type_": "int64"},
                )
            )
        
        for i in ['credit_amount', 'outstanding','number_of_installments_to_pay', 'arreas']:
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": i, "type_": "float64"},
                )
            )

        for i in ['date', 'segment_desc', 'credit_type', 'credit_e_o_m_start_date', 'credit_e_o_m_end_date', 'payment_frequency']:
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": i, "type_": "object"},
                )
            )

    if feature_group == 'funds_features':

        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_of_type",
                kwargs={"column": 'customer_id', "type_": "int64"},
            )
        )
        
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_of_type",
                kwargs={"column": 'funds_balance', "type_": "float64"},
            )
        )

        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_of_type",
                kwargs={"column": 'date', "type_": "object"},
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


def test_data(df, loans, funds):
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

    
    validation_expectation_suite_customer = build_expectation_suite("customer_expectations_raw", "customers_features")
    validation_expectation_suite_loan = build_expectation_suite("loan_expectations_raw", "loans_features")
    validation_expectation_suite_fund = build_expectation_suite("fund_expectations_raw", "funds_features")

    context.add_or_update_expectation_suite(expectation_suite=validation_expectation_suite_customer)
    context.add_or_update_expectation_suite(expectation_suite=validation_expectation_suite_loan)
    context.add_or_update_expectation_suite(expectation_suite=validation_expectation_suite_fund)

    # add data
    logger.info(f"The dataset contains {len(df.columns) + len(loans.columns) + len(funds.columns)} columns.")

    df.columns = [re.sub(r'(?<!^)(?=[A-Z])', '_', col).lower() for col in df.columns]
    df.columns = df.columns.str.replace('.', '', regex=False)
    logger.info(f"{df.columns}")

    loans.columns = [re.sub(r'(?<!^)(?=[A-Z])', '_', col).lower() for col in loans.columns]
    loans.columns = loans.columns.str.replace('.', '', regex=False)
    logger.info(f"{loans.columns}")

    funds.columns = [re.sub(r'(?<!^)(?=[A-Z])', '_', col).lower() for col in funds.columns]
    funds.columns = funds.columns.str.replace('.', '', regex=False)
    logger.info(f"{funds.columns}")

    customer_features = df.columns.tolist()
    loans_features = loans.columns.tolist()
    funds_features = funds.columns.tolist()
    
    df = df.reset_index()
    loans = loans.reset_index()
    funds = funds.reset_index()

    data_asset_name = "customers_raw"
    try:
        data_asset = datasource.add_dataframe_asset(name=data_asset_name, dataframe=df)
    except:
        logger.info("The data asset already exists. The required one will be loaded.")
        data_asset = datasource.get_asset(data_asset_name)

    # Customers
    batch_request = data_asset.build_batch_request(dataframe=df)

    checkpoint_customers = gx.checkpoint.SimpleCheckpoint(
        name="checkpoint_customers_raw",
        data_context=context,
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": "customer_expectations_raw",
            },
        ],
    )

    data_asset_name = "loans_raw"
    try:
        data_asset = datasource.add_dataframe_asset(name=data_asset_name, dataframe=loans)
    except:
        logger.info("The data asset already exists. The required one will be loaded.")
        data_asset = datasource.get_asset(data_asset_name)

    # Loans
    batch_request = data_asset.build_batch_request(dataframe=loans)
    checkpoint_loans = gx.checkpoint.SimpleCheckpoint(
        name="checkpoint_loans_raw",
        data_context=context,
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": "loan_expectations_raw",
            },
        ],
    )
        
    data_asset_name = "funds_raw"
    try:
        data_asset = datasource.add_dataframe_asset(name=data_asset_name, dataframe=funds)
    except:
        logger.info("The data asset already exists. The required one will be loaded.")
        data_asset = datasource.get_asset(data_asset_name)

    # Funds
    batch_request = data_asset.build_batch_request(dataframe=funds)
    checkpoint_funds = gx.checkpoint.SimpleCheckpoint(
        name="checkpoint_funds_raw",
        data_context=context,
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": "fund_expectations_raw",
            },
        ],
    )
    checkpoint_result1 = checkpoint_customers.run()
    checkpoint_result2 = checkpoint_loans.run()
    checkpoint_result3 = checkpoint_funds.run()

    df_validation1 = get_validation_results(checkpoint_result1)
    df_validation2 = get_validation_results(checkpoint_result2)
    df_validation3 = get_validation_results(checkpoint_result3)
    df_validation = pd.concat([df_validation1, df_validation2, df_validation3], ignore_index=True)
    #base on these results you can make an assert to stop your pipeline

    # pd_df_ge = gx.from_pandas(df)

    # assert pd_df_ge.expect_column_values_to_be_of_type("duration", "int64").success == True
    # assert pd_df_ge.expect_column_values_to_be_of_type("marital", "str").success == True
    #assert pd_df_ge.expect_table_column_count_to_equal(23).success == False
    
    logger.info("Data passed on the unit data tests")
    logger.info(f'All raw data tests passed: {df_validation[df_validation.Success == False].empty}')

    return df_validation




