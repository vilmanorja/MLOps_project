"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""
import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder , LabelEncoder


from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]


logger = logging.getLogger(__name__)


"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder , LabelEncoder


# Some function for cleaning customer final  features
def clean_customer_features(X_test: pd.DataFrame) -> pd.DataFrame:
    X_test= X_test.drop(columns="run_date", axis=1)
    return X_test

def feature_preprocessing_test( X_test: pd.DataFrame, preprocessor) -> pd.DataFrame:
    
    log = logging.getLogger(__name__)

    X_test_cleaned = clean_customer_features(X_test)

    X_test_processed = preprocessor.transform(X_test_cleaned)

    try:
        feature_names = preprocessor.get_feature_names_out()
        X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names)
    except:
        X_test_processed = pd.DataFrame(X_test_processed)
    log.info(f"The final dataframe has {len(X_test_processed.columns)} columns.")

    return X_test_processed


