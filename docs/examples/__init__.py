import pandas as pd
import numpy as np
import torch
from sklearn.datasets import make_classification, make_regression
import os


# def generate_simulated_tabular_data(num_samples, num_features):
#     """
#     Generate simulated tabular data with a study_id column and num_features columns of random values

#     Parameters
#     ----------
#     num_samples : int
#         Number of samples to generate
#     num_features : int
#         Number of features to generate

#     Returns
#     -------
#     data : pd.DataFrame
#         Simulated tabular data
#     """
#     data = pd.DataFrame()

#     data["study_id"] = [f"{i}" for i in range(num_samples)]

#     for i in range(num_features):
#         feature_name = f"feature{i + 1}"
#         data[feature_name] = np.random.rand(num_samples)

#     return data


# def add_simulated_data_label(data1, data2, pred_type):
#     """
#     Add a prediction label to two datasets, with the same label for both datasets

#     Parameters
#     ----------
#     data1 : pd.DataFrame
#         First dataset
#     data2 : pd.DataFrame
#         Second dataset
#     pred_type : str
#         Type of prediction label to add, one of 'binary', 'multiclass' or 'regression'

#     Returns
#     -------
#     data1 : pd.DataFrame
#         First dataset with prediction label added
#     data2 : pd.DataFrame
#         Second dataset with prediction label added
#     """
#     num_samples = len(data1)

#     if pred_type == "binary":
#         data1["pred_label"] = np.random.choice(
#             [0, 1], num_samples
#         )  # Example binary prediction label
#     elif pred_type == "multiclass":
#         data1["pred_label"] = np.random.choice(
#             [i for i in range(params["num_classes"])], num_samples
#         )
#     elif pred_type == "regression":
#         data1["pred_label"] = np.random.rand(num_samples)
#     else:
#         raise ValueError(
#             f"pred_type must be one of 'binary', 'multiclass' or 'regression', not {pred_type}"
#         )

#     data2["pred_label"] = data1["pred_label"]

#     data1.set_index("study_id", inplace=True)
#     data2.set_index("study_id", inplace=True)

#     return data1, data2


def generate_simulated_image_data(num_samples, img_dims):
    """
    Generate simulated image data with num_samples samples and img_dims dimensions

    Parameters
    ----------
    num_samples : int
        Number of samples to generate
    img_dims : tuple
        Dimensions of the image data to generate. (num_channels, height, width) or (num_channels, height, width, depth).
        If only 2 dimensions are provided, a 1-dimensional channel will be added to the beginning of the tuple.

    Returns
    -------
    simulated_images : torch.tensor
        Simulated image data
    """

    if len(img_dims) == 2:
        # need to add a 1-dimensional channel to beginning of tuple
        img_dims = (1,) + img_dims
    simulated_images = []
    for _ in range(num_samples):
        simulated_image = np.random.random(img_dims)  # Generate random values
        simulated_image = torch.tensor(simulated_image, dtype=torch.float32)
        simulated_images.append(simulated_image)

    simulated_images = torch.stack(simulated_images)

    return simulated_images


# def generate_all_random_simulated_data(
#     num_samples, num_tab1_features, num_tab2_features, img_dims, params
# ):
#     """
#     Generate simulated data for all modalities, adds a prediction label and returns the params dictionary
#     with the paths to the simulated data, ready to be passed to the dataloader.

#     Parameters
#     ----------
#     num_samples : int
#         Number of samples to generate
#     num_tab1_features : int
#         Number of features to generate for tabular1 data
#     num_tab2_features : int
#         Number of features to generate for tabular2 data
#     img_dims : tuple
#         Dimensions of the image data to generate
#     params : dict
#         Dictionary of parameters

#     Returns
#     -------
#     params : dict
#         Dictionary of parameters with the paths to the simulated data added
#     """
#     params[
#         "tabular1_source"
#     ] = "/Users/florencetownend/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Projects/fusion_library_running/examples/simulated_data/tabular1data.csv"
#     params[
#         "tabular2_source"
#     ] = "/Users/florencetownend/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Projects/fusion_library_running/examples/simulated_data/tabular2data.csv"
#     params[
#         "img_source"
#     ] = "/Users/florencetownend/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Projects/fusion_library_running/examples/simulated_data/imagedata.pt"

#     tabular1_data = generate_simulated_tabular_data(num_samples, num_tab1_features)
#     tabular2_data = generate_simulated_tabular_data(num_samples, num_tab2_features)
#     img_data = generate_simulated_image_data(num_samples, img_dims)

#     tabular1_data, tabular2_data = add_simulated_data_label(
#         tabular1_data, tabular2_data, params["pred_type"]
#     )

#     # save to csv and pt
#     tabular1_data.to_csv(params["tabular1_source"])
#     tabular2_data.to_csv(params["tabular2_source"])
#     torch.save(img_data, params["img_source"])

#     return params


def generate_sklearn_simulated_data(
        num_samples, num_tab1_features, num_tab2_features, img_dims, params
):
    """
    Generate simulated data for all modalities, adds a prediction label and returns the params dictionary

    Parameters
    ----------
    num_samples : int
        Number of samples to generate
    num_tab1_features : int
        Number of features to generate for tabular1 data
    num_tab2_features : int
        Number of features to generate for tabular2 data
    img_dims : tuple
        Dimensions of the image data to generate
    params : dict
        Dictionary of parameters

    Returns
    -------
    params : dict
        Dictionary of parameters with the paths to the simulated data added
    """

    params["tabular1_source"] = "../../../fusilli/utils/simulated_data/tabular1data.csv"
    params["tabular2_source"] = "../../../fusilli/utils/simulated_data/tabular2data.csv"
    params["img_source"] = "../../../fusilli/utils/simulated_data/imagedata.pt"

    if params["pred_type"] == "binary":
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
    elif params["pred_type"] == "multiclass":
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
    elif params["pred_type"] == "regression":
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
            f"pred_type must be one of 'binary', 'multiclass' or 'regression', not {pred_type}"
        )

    tabular1_data = pd.DataFrame()
    tabular1_data["study_id"] = [f"{i}" for i in range(num_samples)]
    for i in range(num_tab1_features):
        feature_name = f"feature{i + 1}"
        tabular1_data[feature_name] = all_tab_features[:, i]
    tabular1_data.set_index("study_id", inplace=True)
    tabular1_data["pred_label"] = labels

    tabular2_data = pd.DataFrame()
    tabular2_data["study_id"] = [f"{i}" for i in range(num_samples)]
    for i in range(num_tab2_features):
        feature_name = f"feature{i + 1}"
        tabular2_data[feature_name] = all_tab_features[:, num_tab1_features + i]
    tabular2_data.set_index("study_id", inplace=True)
    tabular2_data["pred_label"] = labels

    img_data = generate_simulated_image_data(num_samples, img_dims)

    # save to csv and pt
    tabular1_data.to_csv(params["tabular1_source"])
    tabular2_data.to_csv(params["tabular2_source"])
    # torch.save(img_data, params["img_source"])

    return params
