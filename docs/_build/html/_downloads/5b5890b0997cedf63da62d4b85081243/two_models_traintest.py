"""
Train test comparing two models on simulated data
=======================================

This example shows how to train and test two fusion models on simulated data.
"""


# from example_helpers import generate_all_random_simulated_data, generate_sklearn_simulated_data
from docs.source.examples import generate_sklearn_simulated_data
from fusionlibrary.data import get_data_module
from fusionlibrary.train import train_and_save_models
from fusionlibrary.utils.model_chooser import get_models
from fusionlibrary.fusion_models.base_model import BaseModel
from fusionlibrary.eval import Plotter
import os
import torch
from tqdm.auto import tqdm
import importlib

# %matplotlib inline
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*MPS available but not used.*")
warnings.filterwarnings(
    "ignore", message="Checkpoint directory.*exists and is not empty."
)

# %%
# import methods

model_conditions = {
    "class_name": ["ConcatTabularData", "TabularChannelWiseMultiAttention"],
}

imported_models = get_models(model_conditions)
print("Imported methods:")
print(imported_models.method_name.values)

fusion_models = []  # contains the class objects for each model
for index, row in imported_models.iterrows():
    module = importlib.import_module(row["method_path"])
    module_class = getattr(module, row["class_name"])

    fusion_models.append(module_class)


# %%
# training parameters

params = {
    "test_size": 0.2,
    # "subspace_latdims": 5,
    "kfold_flag": False,
    # "num_k": 5,
    # "multiclass_dims": 3,
    "log": False,
    "pred_type": "regression",
}


# %%
# params = generate_all_simulated_data(
#     num_samples=500,
#     num_tab1_features=10,
#     num_tab2_features=10,
#     img_dims=(1, 100, 100),
#     params=params,
# )

params = generate_sklearn_simulated_data(
    num_samples=500,
    num_tab1_features=10,
    num_tab2_features=10,
    img_dims=(1, 100, 100),
    params=params,
)

# %%
all_trained_models = {}  # create dictionary to store trained models

# %% [markdown]
# Training the first fusion model

# %%
# choose model
fusion_model = fusion_models[0]
single_model_dict = {}

# initialise model
init_model = BaseModel(
    fusion_model(
        params["pred_type"], data_dims=[10, 10, [100, 100, 100]], params=params
    )
)


print("method_name:", init_model.method_name)
print("modality_type:", init_model.modality_type)
print("fusion_type:", init_model.fusion_type)
print("metric_name_list:", init_model.metric_names_list)

# get the data module
dm = get_data_module(init_model=init_model, params=params)

# train and test
single_model_dict = train_and_save_models(
    trained_models_dict=single_model_dict,
    data_module=dm,
    params=params,
    fusion_model=fusion_model,
    init_model=init_model,
)

all_trained_models[fusion_model.__name__] = single_model_dict[fusion_model.__name__]


# %%

plotter = Plotter(single_model_dict, params)
single_model_figures_dict = plotter.plot_all()
plotter.show_all(single_model_figures_dict)

# %% [markdown]
# Training tabular decision model

# %%
# choose model
fusion_model = fusion_models[1]
single_model_dict = {}

# initialise model
init_model = BaseModel(
    fusion_model(
        params["pred_type"], data_dims=[10, 10, [100, 100, 100]], params=params
    )
)


print("method_name:", init_model.method_name)
print("modality_type:", init_model.modality_type)
print("fusion_type:", init_model.fusion_type)
print("metric_name_list:", init_model.metric_names_list)

# get the data module
dm = get_data_module(init_model=init_model, params=params)

# train and test
trained_models = train_and_save_models(
    trained_models_dict=single_model_dict,
    data_module=dm,
    params=params,
    fusion_model=fusion_model,
    init_model=init_model,
)

all_trained_models[fusion_model.__name__] = single_model_dict[fusion_model.__name__]


# %%

plotter = Plotter(single_model_dict, params)
single_model_figures_dict = plotter.plot_all()
plotter.show_all(single_model_figures_dict)
# %%
# visualise the results of a single model
comparison_plotter = Plotter(all_trained_models, params)
comparison_plot_dict = comparison_plotter.plot_all()
comparison_plotter.show_all(comparison_plot_dict)

# %%
performances_df = comparison_plotter.save_performance_csv()
performances_df
