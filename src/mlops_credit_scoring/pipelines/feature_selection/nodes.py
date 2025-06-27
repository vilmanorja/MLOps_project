import logging
from typing import Any, Dict, Tuple, List
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def feature_selection(
    X_train: pd.DataFrame, y_train: pd.Series, parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, List[str]]:
    y_train = np.ravel(y_train)

    num_methods = parameters["num_methods"]

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

    # Combine features that appear in at least 'num_methods' methods
    all_features = list(features_kbest | features_lasso | features_rfe)
    counts = Counter(f for fs in [features_kbest, features_lasso, features_rfe] for f in fs)
    features_selected = [f for f in all_features if counts[f] >= num_methods]

    # Build the combined ranking
    ranking_df = pd.DataFrame(index=features_selected)
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
    X_selected = X_train[features_selected]
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

    final_features = [f for f in features_selected if f not in features_to_drop]


    ranking_plot_df = ranking_df.loc[final_features].copy()
    ranking_plot_df = ranking_plot_df.sort_values("combined_rank", ascending=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="combined_rank",
        y=ranking_plot_df.index,
        data=ranking_plot_df,
        palette="viridis"
    )
    plt.title("Feature Importance (Combined Ranking)")
    plt.xlabel("Combined Rank (lower is better)")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()

    return final_features