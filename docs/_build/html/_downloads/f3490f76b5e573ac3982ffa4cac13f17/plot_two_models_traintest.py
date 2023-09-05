"""
Regression: comparing two tabular models trained on simulated data
====================================================================

This script shows how to train two fusion models on a regression task with train/test protocol and multimodal tabular data.

Key Features:

- Importing models based on name.
- Training and testing models with train/test protocol.
- Saving trained models to a dictionary for later analysis.
- Plotting the results of a single model.
- Plotting the results of multiple models as a bar chart.
- Saving the results of multiple models as a csv file.
"""

import importlib

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from docs.examples import generate_sklearn_simulated_data
from fusionlibrary.datamodules import get_data_module
from fusionlibrary.eval_functions import Plotter
from fusionlibrary.fusion_models.base_pl_model import BaseModel
from fusionlibrary.train_functions import train_and_save_models
from fusionlibrary.utils.model_chooser import get_models


# %%
# 1. Import fusion models
# ------------------------
# Here we import the fusion models to be compared. The models are imported using the
# :func:`~fusionlibrary.utils.model_chooser.get_models` function, which takes a dictionary of conditions
# as an input. The conditions are the attributes of the models, e.g. the class name, the modality type, etc.
#
# The function returns a dataframe of the models that match the conditions. The dataframe contains the
# method name, the class name, the modality type, the fusion type, the path to the model, and the path to the
# model's parent class. The paths are used to import the models with the :func:`importlib.import_module`.
#
# We're importing ConcatTabularData and TabularChannelWiseMultiAttention models for this example. Both are multimodal tabular models.

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
# 2. Set the training parameters
# --------------------------------
# Here we define the parameters for training and testing the models. The parameters are stored in a dictionary and passed to most
# of the methods in this library.
# For training and testing, the necessary parameters are:
#
# - ``test_size``: the proportion of the data to be used for testing.
# - ``kfold_flag``: the user sets this to False for train/test protocol.
# - ``log``: a boolean of whether to log the results using Weights and Biases.
# - ``pred_type``: the type of prediction to be performed. This is either ``regression``, ``binary``, or ``classification``. For this example we're using regression.
#
# If we were going to use a subspace-based fusion model, we would also need to set the latent dimensionality of the subspace with ``subspace_latdims``. This will be shown in a different example.

params = {
    "test_size": 0.2,
    "kfold_flag": False,
    "log": False,
    "pred_type": "regression",
}


# %%
# 3. Generating simulated data
# --------------------------------
# Here we generate simulated data for the two tabular modalities for this example.
# This function also simulated image data which we aren't using here.

params = generate_sklearn_simulated_data(
    num_samples=500,
    num_tab1_features=10,
    num_tab2_features=10,
    img_dims=(1, 100, 100),
    params=params,
)

# %%
# 4. Training the first fusion model
# ----------------------------------
# Here we train the first fusion model. We're using the ``train_and_save_models`` function to train and test the models.
# This function takes the following inputs:
#
# - ``trained_models_dict``: a dictionary to store the trained models.
# - ``data_module``: the data module containing the data.
# - ``params``: the parameters for training and testing.
# - ``fusion_model``: the fusion model to be trained.
# - ``init_model``: the initialised dummy fusion model.
#
# First we'll create a dictionary to store both the trained models so we can compare them later.
all_trained_models = {}  # create dictionary to store trained models

# %%
# To train the first model we need to:
#
# 1. *Choose the model*: We're using the first model in the ``fusion_models`` list we made earlier.
# 2. *Create a dictionary to store the trained model*: We're using the name of the model as the key. It may seem overkill to make a dictionary just to store one model, but we also use this when we do k-fold training to store the trained models from the different folds.
# 3. *Initialise the model with dummy data*: This is so we can find out whether there are extra instructions for creating the datamodule (such as a method for creating a graph datamodule).
# 4. *Print the attributes of the model*: To check it's been initialised correctly.
# 5. *Create the datamodule*: This is done with the :func:`~fusionlibrary.datamodules.get_data_module` function. This function takes the initialised model and the parameters as inputs. It returns the datamodule.
# 6. *Train and test the model*: This is done with the :func:`~fusionlibrary.train_functions.train_and_save_models` function. This function takes the trained_models_dict, the datamodule, the parameters, the fusion model, and the initialised model as inputs. It returns the trained_models_dict with the trained model added to it.
# 7. *Add the trained model to the ``all_trained_models`` dictionary*: This is so we can compare the results of the two models later.

fusion_model = fusion_models[0]
single_model_dict = {}

print("Method name:", fusion_model.method_name)
print("Modality type:", fusion_model.modality_type)
print("Fusion type:", fusion_model.fusion_type)

# Create the data module
dm = get_data_module(fusion_model=fusion_model, params=params)

# Train and test
single_model_dict = train_and_save_models(
    trained_models_dict=single_model_dict,
    data_module=dm,
    params=params,
    fusion_model=fusion_model,
)

# Add trained model to dictionary
all_trained_models[fusion_model.__name__] = single_model_dict[fusion_model.__name__]


# %%
# 5. Plotting the results of the first model
# --------------------------------------------
# We're using the :class:`~fusionlibrary.eval_functions.Plotter` class to plot the results of the first model. This class takes the dictionary of trained models and the parameters as inputs. It returns a dictionary of figures.
# If there is one model in the dictionary (i.e. only one unique key), then it plots the figures for analysing the results of a single model.

plotter = Plotter(single_model_dict, params)
single_model_figures_dict = plotter.plot_all()
plotter.show_all(single_model_figures_dict)

# %% [markdown]
# 6. Training the second fusion model
# -------------------------------------
# Here we train the second fusion model: TabularChannelWiseMultiAttention. We're using the same steps as before, but this time we're using the second model in the ``fusion_models`` list.


# %%
# Choose the model
fusion_model = fusion_models[1]
single_model_dict = {}


print("Method name:", fusion_model.method_name)
print("Modality type:", fusion_model.modality_type)
print("Fusion type:", fusion_model.fusion_type)

# Create the data module
dm = get_data_module(fusion_model=fusion_model, params=params)

# Train and test
trained_models = train_and_save_models(
    trained_models_dict=single_model_dict,
    data_module=dm,
    params=params,
    fusion_model=fusion_model,
)

# Add trained model to dictionary
all_trained_models[fusion_model.__name__] = single_model_dict[fusion_model.__name__]


# %%
# 7. Plotting the results of the second model
# ----------------------------------------------

plotter = Plotter(single_model_dict, params)
single_model_figures_dict = plotter.plot_all()
plotter.show_all(single_model_figures_dict)

# %%
# 8. Comparing the results of the two models
# ---------------------------------------------
# Now we're going to compare the results of the two models. We're using the same steps as when we used Plotter before, but this time we're using the ``all_trained_models`` dictionary which contains both models.

comparison_plotter = Plotter(all_trained_models, params)
comparison_plot_dict = comparison_plotter.plot_all()
comparison_plotter.show_all(comparison_plot_dict)

# %%
# 9. Saving the metrics of the two models
# -----------------------------------------
# We can also get the metrics of the two models into a Pandas DataFrame using the :func:`~fusionlibrary.eval_functions.Plotter.get_performance_df` function.
performances_df = comparison_plotter.get_performance_df()
performances_df
