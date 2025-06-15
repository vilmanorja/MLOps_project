# streamlit libray
import streamlit as st
import pickle
import sys
import pandas as pd
from kedro.io import DataCatalog
from streamlit_ydata_profiling import st_profile_report

st.title('Data Catalog from Kedro')


#----------------------------------------------------------------------------
# KEDRO CONFIG

from pathlib import Path
from kedro.framework.project import configure_project

package_name = Path(__file__).parent.name
configure_project(package_name)

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

conf_path = str(Path('..') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
catalog = conf_loader["catalog"]

if 'button_model' not in st.session_state:
    st.session_state.button_model = False

def button_model():
    st.session_state.button_model = True 

def stateful_button(*args, key=None, **kwargs):
    if key is None:
        raise ValueError("Must pass key")

    if key not in st.session_state:
        st.session_state[key] = False

    if st.button(*args, **kwargs):
        st.session_state[key] = not st.session_state[key]

    return st.session_state[key]    

# Initialization
if 'use_case' not in st.session_state:
    st.session_state['use_case'] = ''


choice = st.radio('**Available Dataset:**',list(catalog.keys()), index= 0)
st.write("Important: Do not try to generate a report on a non csv object of the Data Catalog")
from kedro.io import DataCatalog

catalog[choice]["filepath"] = "../" + catalog[choice]["filepath"]

datacatalog = DataCatalog.from_config(catalog)

dataset = datacatalog.load(choice)

from ydata_profiling import ProfileReport

profile = ProfileReport(dataset, title=f"Bank Profiling Report", minimal=True)

if choice != '':
    if stateful_button(f'**Generate report on data**', key=f'**Generate report on data**'):
        st_profile_report(profile, navbar=True)