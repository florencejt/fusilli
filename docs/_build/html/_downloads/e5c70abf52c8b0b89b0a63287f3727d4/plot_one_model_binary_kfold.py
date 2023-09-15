"""
Binary: training one kfold model
==================================================================

This script shows how to train one fusion models on a binary task with k-fold training protocol and multimodal tabular data.

Key Features:

- Importing a model based on its path.
- Training and testing a model with k-fold cross validation.
- Plotting the results of a single k-fold model.
"""

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# %%
from docs.examples import generate_sklearn_simulated_data
from fusionlibrary.data import get_data_module
from fusionlibrary.eval import Plotter
from fusionlibrary.fusion_models.base_model import BaseModel

# %%
# 1. Import model
# --------------------
from fusionlibrary.fusion_models.tab_crossmodal_att import (
    TabularCrossmodalMultiheadAttention,
)
from fusionlibrary.train import train_and_save_models

# %%
# 2. Set the training parameters
# --------------------------------

params = {
    "test_size": 0.2,
    "kfold_flag": True,
    "num_k": 5,
    "log": False,
    "pred_type": "binary",
    "batch_size": 32,
}

# %%
# 3. Generate simulated data
# ----------------------------
params = generate_sklearn_simulated_data(
    num_samples=500,
    num_tab1_features=10,
    num_tab2_features=10,
    img_dims=(1, 100, 100),
    params=params,
)

# %%
# 4. Initialise model
fusion_model = TabularCrossmodalMultiheadAttention

single_model_dict = {}

print("method_name:", fusion_model.method_name)
print("modality_type:", fusion_model.modality_type)
print("fusion_type:", fusion_model.fusion_type)

# %%
# 5. Train and test the model
# ----------------------------
dm = get_data_module(
    fusion_model=fusion_model, params=params, batch_size=params["batch_size"]
)

# train and test
single_model_dict = train_and_save_models(
    trained_models_dict=single_model_dict,
    data_module=dm,
    params=params,
    fusion_model=fusion_model,
)

# %%
# 6. Plot the results
# ----------------------------
plotter = Plotter(single_model_dict, params)
single_model_figures_dict = plotter.plot_all()
plotter.show_all(single_model_figures_dict)
