import pandas as pd
from sklearn.datasets import make_classification, make_regression


def generate_sklearn_simulated_data(
        prediction_task, num_samples, num_tab1_features, num_tab2_features, external=False
):
    """
    Generate simulated data for all modalities, adds a prediction label and returns the params dictionary

    Parameters
    ----------
    prediction_task : str
        The type of prediction to be performed. This is either ``regression``, ``binary``, or ``classification``.
    num_samples : int
        Number of samples to generate
    num_tab1_features : int
        Number of features to generate for tabular1 data
    num_tab2_features : int
        Number of features to generate for tabular2 data

    Returns
    -------
    params : dict
        Dictionary of parameters with the paths to the simulated data added
    """
    if external:
        tabular1_path = "../../../fusilli/utils/simulated_data/external_tabular1data.csv"
        tabular2_path = "../../../fusilli/utils/simulated_data/external_tabular2data.csv"
    else:
        tabular1_path = "../../../fusilli/utils/simulated_data/tabular1data.csv"
        tabular2_path = "../../../fusilli/utils/simulated_data/tabular2data.csv"

    if prediction_task == "binary":
        # Creating a simulated feature matrix and output vector with 100 samples
        all_tab_features, labels = make_classification(
            n_samples=num_samples,
            n_features=num_tab1_features + num_tab2_features,  # taking  features
            n_informative=(num_tab1_features + num_tab2_features)
                          // 3,  # features that predict the output's classes
            n_classes=2,  # three output classes
            weights=None,  # equal number of samples per class)
            flip_y=0.1,  # flip 10% of the labels
        )
    elif prediction_task == "multiclass":
        num_classes = 3
        all_tab_features, labels = make_classification(
            n_samples=num_samples,
            n_features=num_tab1_features + num_tab2_features,  # taking  features
            n_informative=(num_tab1_features + num_tab2_features)
                          // 2,  # features that predict the output's classes
            n_classes=num_classes,  # three output classes
            weights=None,  # equal number of samples per class)
            flip_y=0.1,  # flip 10% of the labels
        )
    elif prediction_task == "regression":
        all_tab_features, labels = make_regression(
            n_samples=num_samples,
            n_features=num_tab1_features + num_tab2_features,  # taking  features
            n_informative=(num_tab1_features + num_tab2_features)
                          // 2,  # features that predict the output's classes
            noise=3,
            effective_rank=3,
        )
    else:
        raise ValueError(
            f"pred_type must be one of 'binary', 'multiclass' or 'regression', not {prediction_task}"
        )

    tabular1_data = pd.DataFrame()
    tabular1_data["ID"] = [f"{i}" for i in range(num_samples)]
    for i in range(num_tab1_features):
        feature_name = f"feature{i + 1}"
        tabular1_data[feature_name] = all_tab_features[:, i]
    tabular1_data.set_index("ID", inplace=True)
    tabular1_data["prediction_label"] = labels

    tabular2_data = pd.DataFrame()
    tabular2_data["ID"] = [f"{i}" for i in range(num_samples)]
    for i in range(num_tab2_features):
        feature_name = f"feature{i + 1}"
        tabular2_data[feature_name] = all_tab_features[:, num_tab1_features + i]
    tabular2_data.set_index("ID", inplace=True)
    tabular2_data["prediction_label"] = labels

    # save to csv and pt
    tabular1_data.to_csv(tabular1_path)
    tabular2_data.to_csv(tabular2_path)

    return tabular1_path, tabular2_path
