import logging
from typing import Any, Dict, Tuple

from great_expectations.core import ExpectationSuite, ExpectationConfiguration
#from great_expectations.core.expectation_suite import ExpectationSuite
#from great_expectations.expectations.expectation_configuration import ExpectationConfiguration

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
    

    # customer features
    if feature_group == 'customers_features':

        for i in ['NoOfDependents', 'SegmentId', 'IndustryId', 'LegalDocName1Id', 'YrNetMonthlyIn']:
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
                kwargs={"column": "NewId"}
            )
        )
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_of_type",
                kwargs={"column": "NewId", "type_": "int64"}
            )
        )

        for i in ['SegmentId', 'IndustryId', 'LegalDocName1Id', 'NewId']:
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={
                        "column": i,
                        "min_value": 1,
                        "strict_min": False,
                        "max_value": None
                    }
                )
            )

        for i in ['YrNetMonthlyIn', 'NoOfDependents']:
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_between",
                    kwargs={
                        "column": i,
                        "min_value": 0,
                        "strict_min": False,
                        "max_value": None
                    }
                )
            )

        for i in ['YrNetMonthlyIn', 'NoOfDependents', 'SegmentId', 'IndustryId', 'LegalDocName1Id', 'NewId']:
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": i}
                )
            )

        for i in ['CustomerStatus', 'EmploymentStatus', 'Gender', 'MaritalStatus', 'Placebrth', 'CustType', 
                'Nationality', 'OcupationDesc', 'ResidenceCode', 'ResidenceStatus', 'ResidenceType',
                'SegGroup', 'Title', 'TownCountry', 'CustType.1', 'Habliter', 'Province', 'District', 
                'LegalDocName1IdDescription', 'LegalIssDate', 'LegalIssAuth', 'AMLRiskRating']:
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": i}
                )
            )

        object_features = ['CustomerSince', 'CustomerStatus', 'DateOfBirth', 'EmploymentStatus',
       'Gender', 'MaritalStatus', 'Placebrth', 'CustType', 'Nationality',
       'OcupationDesc', 'ResidenceCode', 'ResidenceStatus', 'ResidenceType',
       'SegGroup', 'Title', 'TownCountry', 'BirthInCorpDate', 'CustType.1',
       'Habliter', 'Province', 'District', 'LegalDocName1IdDescription',
       'LegalIssDate', 'LegalExpDate', 'LegalIssAuth', 'AMLRiskRating']

        for i in object_features:
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_to_exist",
                    kwargs={"column": i}
                )
            )

        for i in object_features:
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": i, "type_": "object"}
                )
            )


    if feature_group == 'loans_features':

        for i in ['CustomerNewId', 'ContractId', 'HasDefault']:
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": i, "type_": "int64"},
                )
            )
        
        for i in ['CreditAmount', 'Outstanding', 'NumberOfInstallmentsToPay', 'Arreas']:
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": i, "type_": "float64"},
                )
            )

        for i in ['SegmentDesc', 'CreditType', 'CreditEOMStartDate', 'CreditEOMEndDate', 'PaymentFrequency']:
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
                kwargs={"column": 'CustomerId', "type_": "int64"},
            )
        )
        
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_of_type",
                kwargs={"column": 'FundsBalance', "type_": "float64"},
            )
        )

        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_of_type",
                kwargs={"column": 'Date', "type_": "object"},
            )
        )

    if feature_group == 'transactions_features':
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_values_to_be_of_type",
                kwargs={"column": 'TransactionId', "type_": "int64"},
            )
        )

        for i in ['CustomerIdDebitNew', 'CustomerIdCreditNew', 'Amount', 'AmountMZN']:
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": i, "type_": "float64"},
                )
            )

        for i in ['Date', 'TransactionType', 'TransactionCategory', 'Currency']:
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": i, "type_": "object"},
                )
            )

    if feature_group == 'loans_hist_features':
        for i in ['CustomerNewId', 'ContractId', 'HasDefault']:
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": i, "type_": "int64"},
                )
            )
        for i in ['CreditAmount', 'Outstanding', 'Arreas']:
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": i, "type_": "float64"},
                )
            )
        for i in ['SnapshotDate', 'SegmentDesc', 'CreditType', 'CreditEOMStartDate', 'CreditEOMEndDate', 'NumberOfInstallmentsToPay', 'PaymentFrequency']:
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": i, "type_": "object"},
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


def test_data(df, funds, transactions, loans_files, loans_hist, run_date):
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
    validation_expectation_suite_funds = build_expectation_suite("funds_expectations_raw", "funds_features")
    validation_expectation_suite_transactions = build_expectation_suite("transactions_expectations_raw", "transactions_features")
    validation_expectation_suite_loans = build_expectation_suite("loans_expectations_raw", "loans_features")
    validation_expectation_suite_loans_hist = build_expectation_suite("loans_hist_expectations_raw", "loans_hist_features")

    context.add_or_update_expectation_suite(expectation_suite=validation_expectation_suite_customer)
    context.add_or_update_expectation_suite(expectation_suite=validation_expectation_suite_funds)
    context.add_or_update_expectation_suite(expectation_suite=validation_expectation_suite_transactions)
    context.add_or_update_expectation_suite(expectation_suite=validation_expectation_suite_loans)
    context.add_or_update_expectation_suite(expectation_suite=validation_expectation_suite_loans_hist)

    # add data
    for date in run_date:
        key = f"Loans_{date}"  # No ".csv" suffix here
        dataset = loans_files.get(key)
        if dataset is None:
            print(f"{key} → SKIPPED")
            continue
        loans = dataset()

    # logger.info(f"The dataset contains {len(df.columns)} columns.")
    # logger.info(f"The dataset contains {len(loans.columns)} columns.")
    # logger.info(f"The dataset contains {len(funds.columns)} columns.")
    df = df.reset_index()
    funds = funds.reset_index()
    transactions = transactions.reset_index()
    loans = loans.reset_index()
    loans_hist = loans_hist.reset_index()

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
                "expectation_suite_name": "loans_expectations_raw",
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
                "expectation_suite_name": "funds_expectations_raw",
            },
        ],
    )

    data_asset_name = "transactions_raw"
    try:
        data_asset = datasource.add_dataframe_asset(name=data_asset_name, dataframe=transactions)
    except:
        print("The data asset alread exists. The required one will be loaded.")
        data_asset = datasource.get_asset(data_asset_name)
    batch_request = data_asset.build_batch_request(dataframe=transactions)

    checkpoint_transactions = gx.checkpoint.SimpleCheckpoint(
            name="checkpoint_transactions_raw",
            data_context=context,
            validations=[
                {
                    "batch_request": batch_request,
                    "expectation_suite_name": "transactions_expectations_raw",
                },
            ],
    )
    data_asset_name = "loans_hist_raw"
    try:
        data_asset = datasource.add_dataframe_asset(name=data_asset_name, dataframe=loans_hist)
    except:
        print("The data asset alread exists. The required one will be loaded.")
        data_asset = datasource.get_asset(data_asset_name)
    batch_request = data_asset.build_batch_request(dataframe=loans_hist)

    checkpoint_loans_hist = gx.checkpoint.SimpleCheckpoint(
            name="checkpoint_loans_hist_raw",
            data_context=context,
            validations=[
                {
                    "batch_request": batch_request,
                    "expectation_suite_name": "loans_hist_expectations_raw",
                },
            ],
    )

    checkpoint_result1 = checkpoint_customers.run()
    checkpoint_result2 = checkpoint_loans.run()
    checkpoint_result3 = checkpoint_funds.run()
    checkpoint_result4 = checkpoint_transactions.run()
    checkpoint_result5 = checkpoint_loans_hist.run()

    df_validation1 = get_validation_results(checkpoint_result1)
    df_validation2 = get_validation_results(checkpoint_result2)
    df_validation3 = get_validation_results(checkpoint_result3)
    df_validation4 = get_validation_results(checkpoint_result4)
    df_validation5 = get_validation_results(checkpoint_result5)
    df_validation = pd.concat([df_validation1, df_validation2, df_validation3, df_validation4, df_validation5], ignore_index=True)
    #base on these results you can make an assert to stop your pipeline

    # pd_df_ge = gx.from_pandas(df)

    # assert pd_df_ge.expect_column_values_to_be_of_type("duration", "int64").success == True
    # assert pd_df_ge.expect_column_values_to_be_of_type("marital", "str").success == True
    # assert pd_df_ge.expect_table_column_count_to_equal(23).success == False
    
    logger.info("Data passed on the unit data tests")
    logger.info(f'All raw data tests passed: {df_validation[df_validation.Success == False].empty}')

    return df_validation, df, funds, transactions, loans_files, loans_hist 




