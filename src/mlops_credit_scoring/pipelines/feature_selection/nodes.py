import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import os
import pickle


# def feature_selection( X_train: pd.DataFrame , y_train: pd.DataFrame,  parameters: Dict[str, Any]):

#     log = logging.getLogger(__name__)
#     log.info(f"We start with: {len(X_train.columns)} columns")

#     if parameters["feature_selection"] == "rfe":
#         y_train = np.ravel(y_train)
#         # open pickle file with regressors
#         try:
#             with open(os.path.join(os.getcwd(), 'data', '06_models', 'champion_model.pkl'), 'rb') as f:
#                 classifier = pickle.load(f)
#         except:
#             classifier = RandomForestClassifier(**parameters['baseline_model_params'])

#         rfe = RFE(classifier) 
#         rfe = rfe.fit(X_train, y_train)
#         f = rfe.get_support(1) #the most important features
#         X_cols = X_train.columns[f].tolist()

#     log.info(f"Number of best columns is: {len(X_cols)}")
    
#     return X_cols


#----------
import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

def feature_selection(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[pd.DataFrame, List[str]]:
    y_train = np.ravel(y_train)

    #X_train= X_train.drop(columns=["num__CustomerId"])

    # SelectKBest
    selector_kbest = SelectKBest(score_func=f_classif, k=15)
    selector_kbest.fit(X_train, y_train)
    features_kbest = set(X_train.columns[selector_kbest.get_support()])
    f_scores = dict(zip(X_train.columns, selector_kbest.scores_))

    # Lasso
    lasso = LassoCV(cv=5, random_state=42, max_iter=5000).fit(X_train, y_train)
    selector_lasso = SelectFromModel(lasso, prefit=True)
    features_lasso = set(X_train.columns[selector_lasso.get_support()])
    lasso_coefs = dict(zip(X_train.columns, np.abs(lasso.coef_)))

    # RFE with RandomForest
    classifier = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rfe = RFE(classifier)
    rfe.fit(X_train, y_train)
    features_rfe = set(X_train.columns[rfe.get_support()])
    rfe_importances = dict(zip(X_train.columns, rfe.estimator_.feature_importances_))

    # Combine features that appear in at least 2 methods
    all_features = list(features_kbest | features_lasso | features_rfe)
    counts = Counter(f for fs in [features_kbest, features_lasso, features_rfe] for f in fs)
    features_2_of_3 = [f for f in all_features if counts[f] >= 2]

    # Build combined ranking
    ranking_df = pd.DataFrame(index=features_2_of_3)
    ranking_df["f_score"] = ranking_df.index.map(f_scores)
    ranking_df["lasso_coef"] = ranking_df.index.map(lasso_coefs)
    ranking_df["rfe_importance"] = ranking_df.index.map(rfe_importances)
    ranking_df = ranking_df.fillna(0)

    ranking_df["combined_rank"] = (
        ranking_df["f_score"].rank(ascending=False) +
        ranking_df["lasso_coef"].rank(ascending=False) +
        ranking_df["rfe_importance"].rank(ascending=False)
    )

    # Correlation filtering
    X_selected = X_train[features_2_of_3]
    corr_matrix = X_selected.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    features_to_drop = set()
    for col1 in upper_tri.columns:
        for col2 in upper_tri.index:
            if pd.notna(upper_tri.loc[col2, col1]) and upper_tri.loc[col2, col1] > 0.7:
                if col1 in features_to_drop or col2 in features_to_drop:
                    continue
                rank1 = ranking_df.loc[col1, "combined_rank"]
                rank2 = ranking_df.loc[col2, "combined_rank"]
                if rank1 < rank2:
                    features_to_drop.add(col1)
                else:
                    features_to_drop.add(col2)

    final_features = [f for f in features_2_of_3 if f not in features_to_drop]

    #final_features.append('num__CustomerId')
    
    print('-------------------------')
    print(f"Selected features: {final_features}")

    return final_features
