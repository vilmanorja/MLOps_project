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

    # categorical features
    if feature_group == 'categorical_features':

        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "customer_status", "value_set": ['Private Client - Standard', 'Corporate - Small', 'Corporate - Medium', 'Private Client High Networth', 'Corporate - Large', 'Financial - Large', 'Financial - Small', 'Proprietorship Standard', 'T24 Updates', 'Deceased Individual', 'Partnership firm Standard', 'Financial - Medium', 'Partnership High Networth', 'Customer Deletion', 'Governmental', 'Hotlisted', 'Proprietorship Highnetworth']},
            )
        ) 
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "employment_status", "value_set": ['MB', 'LP', 'OTHER', 'EMPLOYED', 'TPE', 'UNEMPLOYED', 'SELF-EMPLOYED', 'RET', 'UE', 'STUDENT', 'RETIRED']},
            )
        ) 
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "gender", "value_set": ['FEMALE', 'MALE']},
            )
        ) 
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "marital_status", "value_set": ['OTHER', 'DIVORCED', 'SINGLE', 'MARRIED', 'PARTNER', 'WIDOWED']},
            )
        ) 
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "cust_type", "value_set": ['RETAIL', 'CORPORATE', 'PROSPECT']},
            )
        ) 
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "nationality", "value_set": ['Portugal', 'Mozambique', 'India', 'Pakistan', 'Peoples Republic of China', 'Tokelau', 'South Africa', 'Turkey', 'Netherlands', 'Egypt', 'Switzerland', 'United States of America', 'Congo', 'Yemen', 'Uruguay', 'Afghanistan', 'Korea  Republic of', 'Australia', 'Morocco', 'Somalia', 'Spain', 'Bahrain', 'Lebanon', 'Kenya', 'Mauritius', 'Tanzania  United Republic of', 'Ethiopia', 'Guinea-Bissau', 'Mali', 'Mauritania', 'Guinea', 'Gambia', 'Sierra Leone', 'Mexico', 'Bangladesh', 'Aruba', 'United Arab Emirates', 'Zambia', 'Saudi Arabia', 'Singapore', 'Iran (Islamic Republic of)', 'Nigeria', 'Angola', 'Rwanda', 'Zimbabwe', 'Senegal', 'Hong Kong', 'New Zealand', 'Brazil', 'Italy', 'Cape Verde', 'Philippines', 'Malta', 'Oman', 'Jordan', 'Syrian Arab Republic', 'Libyan Arab Jamahiriya', 'Estado da Palestina', 'Malawi', 'Iraq', 'Great Britain', 'Turks and Caicos Islands', 'Germany', 'Japan', 'Namibia', 'Chile', 'Swaziland', 'Uzbekistan', 'France', 'Luxembourg', 'Norway', 'Bulgaria', 'Croatia', 'Canada', 'Ireland', 'Russian Federation', 'Burundi', 'Czech Republic', 'Uganda', 'Liberia', 'Poland', 'Belgium', 'Benin', 'Greece', 'Kyrgyzstan', 'American Samoa', 'Sao Tome and Principe', 'Tunisia', 'Haiti', 'Columbia', 'Macau', 'Romania', 'Cuba', 'Peru', 'Republic of China (Taiwan)', 'Kazakstan', 'Ghana', 'Finland', 'Puerto Rico', 'Vietnam', 'Israel', 'Thailand', 'Latvia', 'Sri Lanka', 'Nicaragua', 'Congo  Democratic Republic of the', 'Panama', 'Austria', 'Botswana', 'Indonesia', 'Madagascar', 'Comoro Islands', 'Cameroon', 'Nepal', 'Bhutan', 'Sudan', 'Sweden', 'Serbia', 'Mongolia', 'Paraguay', 'Argentina', 'Malaysia', 'Cyprus', 'Honduras', 'Guatemala', 'Monaco', 'Gabon', 'Denmark', 'Eritrea', 'Armenia', 'Ecuador', 'EU Countries', 'Venezuela', 'Dominica', 'Slovenia', 'Antigua And Barbuda', 'Lesotho', 'Georgia', 'Ivory Coast', 'Europa', 'Timor-leste', 'Cayman Islands', 'Northern Mariana Islands', 'San Marino', 'Dominican Republic', 'Chad', 'Algeria', 'Korea  Democratic Peoples Rep. of', 'Iceland', 'Turkmenistan', 'Ukraine', 'Bahamas', 'Bolivia', 'Niger', 'Liechtenstein', 'Hungary', 'Monserrat', 'Jersey', 'Estonia', 'Guadeloupe', 'Costa Rica', 'Lithuania', 'Guyana', 'Qatar', 'Andorra', 'Reunion', 'Yugoslavia', 'Belarus', 'Slovakia', 'Saint Lucia', 'Papua New Guinea', 'Barbados', 'Azerbaijan', 'Fiji', 'Holy See (Vatican City State)', 'Central African Republic']},
            )
        ) 
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "residence_code", "value_set": ['MZ', 'PT', 'ZA', 'NL', 'EG', 'CH', 'CN', 'US', 'IN', 'YE', 'UY', 'AF', 'KR', 'AU', 'ES', 'BH', 'LB', 'MU', 'TZ', 'KE', 'AW', 'AE', 'TR', 'AO', 'NG', 'ZM', 'RW', 'BR', 'ZW', 'MA', 'MR', 'IT', 'NZ', 'MT', 'OM', 'JP', 'CL', 'SZ', 'SA', 'FR', 'LU', 'HR', 'PL', 'GR', 'KG', 'ST', 'TN', 'IR', 'IE', 'GB', 'MW', 'VN', 'JO', 'FI', 'SE', 'PK', 'LV', 'BE', 'CV', 'CU', 'PA', 'MX', 'CO', 'DE', 'AT', 'BW', 'ID', 'MG', 'CA', 'IQ', 'HK', 'KM', 'CM', 'NO', 'CD', 'SG', 'RO', 'RS', 'GH', 'PY', 'PH', 'TW', 'LS', 'MY', 'BM', 'CZ', 'MC', 'AR', 'DK', 'PE', 'XE', 'SI', 'AG', 'TH', 'UG', 'RE', 'GE', 'GM', 'CI', 'KZ', 'CY', 'GA', 'GN', 'MO', 'KP', 'TC', 'IL', 'MN', 'BS', 'UA', 'RU', 'MP', 'HU', 'LR', 'EE', 'AM', 'CR', 'BD', 'SO', 'AD', 'SN', 'SK', 'LC', 'AS', 'BB', 'LT', 'BI', 'LK', 'ML', 'ET', 'EU', 'ER']},
            )
        ) 
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "residence_status", "value_set": ['HOME.OWNER', 'OTHER', 'TENANT', 'LIVING.WTH.PARENTS', 'SQUATTER']},
            )
        ) 
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "residence_type", "value_set": ['RESIDENTIAL.APT', 'INDEPEDENT.HOUSE', 'FARM.HOUSE', 'SERVICED.APT']},
            )
        ) 
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "seg_group", "value_set": ['Company', 'Personal']},
            )
        ) 
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "title", "value_set": ['MRS', 'MR', 'DR', 'DRS', 'MISS', 'MAST1', 'ENG', 'PHD', 'MAST']},
            )
        ) 
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "cust_type1", "value_set": ['RETAIL', 'CORPORATE', 'PROSPECT']},
            )
        ) 
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "habliter", "value_set": ['Ate12 ano', 'Curso Superior', 'Bacharelato', 'Doutoramento', 'Mestrado', 'S.Estudos', 'Ensino Primario']},
            )
        ) 
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "province", "value_set": ['NAMPULA', 'CIDADE DE MAPUTO', 'MAPUTO', 'SOFALA', 'TETE', 'CABO DELGADO', 'NOT APPLICABLE', 'MANICA', 'ZAMBEZIA', 'NIASSA', 'INHAMBANE', 'GAZA']},
            )
        ) 
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "legal_doc_name1_id_description", "value_set": ['Licenca/Alvara', 'DIRE', 'Certidao da Conservatoria de Regist', 'Licenca / Alvara para o exercicio d', 'BI', 'Certidao de reserva de nome', 'Passaporte', 'Others (specify on free text field)', 'Certidao de Registo Comercial (Caso', 'Acta(s) da Assembleia Geral dos Acc', 'Escritura Publica da constituicao d', 'Certidao de Registo / Ministerio da', 'NUIT- Numero Unico de Identificacao', 'Cedula Pessoal', 'Recibo de pedido de BI', 'Cartao de identificacao de refugiad', 'NUIT-Numero Unico de Identificac', 'Carta de conducao', 'BR com o Pacto Social / Estatutos d', 'Termo de Autorizacao do Ministerio"', 'Certidao narrativa completa de nasc', 'Certidao da sentenca do Tribunal de', 'Numero de Registo Fiscal ( Modelo 4', 'Cartao de recenseamento eleitoral', 'Prova de Residencia', 'Prova documental de nao residente', 'Cartao de agricultor/trabalhador ab', 'Cartao de INSS', 'Documento de Transitario', 'Declaracao Comprovativa / Ministeri', 'Cedula militar', 'Documento de Regime Especial', 'Contrato de Trabalho', 'Documento de Embaixada', 'Procuracoes (especificar)', 'Documento de Staff Embaixada', 'Acta(s) da reuniao dos socios (pode', 'Documeto de Exportador']},
            )
        ) 
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                expectation_type="expect_column_distinct_values_to_be_in_set",
                kwargs={"column": "a_m_l_risk_rating", "value_set": ['Elevado', 'Medio', 'Baixo']},
            )
        ) 
        for i in ['placebrth', 'ocupation_desc', 'town_country', 'district', 'legal_iss_auth', 'legal_exp_date', 'date_of_birth', 'birth_in_corp_date', 'legal_iss_date', 'customer_since']:
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




