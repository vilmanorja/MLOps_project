import pandas as pd
import nannyml as nml
import matplotlib.pyplot as plt
import os


def combine_prediction_outputs(X_test, y_preds, y_pred_proba, y_test):
    y_real = y_test.rename(columns={"HasDefault": "HasDefault_real"})
    y_pred = y_preds.rename(columns={"HasDefault": "HasDefault_pred"})
    y_pred_proba = y_pred_proba.rename(columns={"HasDefault": "HasDefault_pred_proba"})

    df = pd.concat([X_test, y_pred, y_pred_proba, y_real], axis=1)
    df["run_date"] = pd.to_datetime(df["run_date"], format="%Y%m%d")
    return df

def detect_data_drift(customers_pred_test: pd.DataFrame, reference_split_date: str):
    reference = customers_pred_test[customers_pred_test["run_date"] <= reference_split_date].copy()
    analysis = customers_pred_test[customers_pred_test["run_date"] > reference_split_date].copy()

    categorical = ['SegGroup', 'AMLRiskRating']
    numerical = ['Avg_Monthly_Income', 'Avg_Monthly_expenses', 'CreditAmount', 'Age']

    univariate_calculator = nml.UnivariateDriftCalculator(
        column_names=numerical + categorical,
        treat_as_categorical=categorical,
        timestamp_column_name='run_date',
        chunk_size=None,
        categorical_methods=["jensen_shannon"],
        continuous_methods=["kolmogorov_smirnov"],
        thresholds={
            "jensen_shannon": nml.thresholds.ConstantThreshold(upper=0.1),
            "kolmogorov_smirnov": nml.thresholds.ConstantThreshold(upper=0.2),
        },
    )

    univariate_calculator.fit(reference)
    results = univariate_calculator.calculate(analysis)
    
    # Save individual plots
    plot_dir = "data/08_reporting/plots/univariate_drift"
    os.makedirs(plot_dir, exist_ok=True)

    for column in numerical + categorical:
        fig = results.filter(column_names=[column], period="analysis").plot(kind="drift")
        fig.write_html(f"{plot_dir}/{column}_drift.html")
        #plt.close(fig)

    return results.filter(period="analysis").to_df()

def detect_multivariate_drift(customers_pred_test: pd.DataFrame, reference_split_date: str):
    reference = customers_pred_test[customers_pred_test["run_date"] <= reference_split_date].copy()
    analysis = customers_pred_test[customers_pred_test["run_date"] > reference_split_date].copy()

    non_feature_columns = ["run_date", "HasDefault_pred", "HasDefault_real", "HasDefault_pred_proba"]
    feature_columns = [col for col in customers_pred_test.columns if col not in non_feature_columns]

    drift_calc = nml.DataReconstructionDriftCalculator(
        column_names=feature_columns,
        timestamp_column_name="run_date",
        chunk_size=None
    )
    drift_calc.fit(reference)
    results = drift_calc.calculate(analysis)
     # Save the figure manually
    plot_dir = "data/08_reporting/plots"
    os.makedirs(plot_dir, exist_ok=True)
    fig = results.plot()
    fig.write_html(f"{plot_dir}/multivariate_drift_plot.html")
    #plt.close(fig)
    # fig.write_image(
    #     f"{plot_dir}/multivariate_drift_plot.png",
    #     width=600,           # smaller width in pixels
    #     height=400,          # smaller height
    #     scale=1              # no scaling (default is 1), lower = faster
    # )
    return results.filter(period="analysis").to_df()

def estimate_performance(customers_pred_test: pd.DataFrame, reference_split_date: str):
    reference = customers_pred_test[customers_pred_test["run_date"] <= reference_split_date].copy()
    analysis = customers_pred_test[customers_pred_test["run_date"] > reference_split_date].copy()

    estimator = nml.CBPE(
        y_true='HasDefault_real',
        y_pred='HasDefault_pred',
        y_pred_proba='HasDefault_pred_proba',
        timestamp_column_name='run_date',
        metrics=['accuracy', 'recall', 'f1'],
        chunk_size=None,
        problem_type='classification_binary',
        thresholds={"accuracy": nml.thresholds.ConstantThreshold(lower=0.60, upper=0.70)}
    )
    estimator = estimator.fit(reference)
    estimated_performance = estimator.estimate(analysis)
    # Save plot manually
    fig = estimated_performance.filter(metrics=['accuracy'], period='analysis').plot()
    output_path = "data/08_reporting/plots/estimated_performance_accuracy.html"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)


    return estimated_performance.filter(period="analysis").to_df()
