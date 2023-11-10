"""
📈 Regression: Comparing Two Tabular Models Trained on Simulated Data 📊
========================================================================

🚀 Welcome to this tutorial on training and comparing two fusion models on a regression task using simulated multimodal tabular data! 🎉

🌟 Key Features:

- 📥 Importing models based on name.
- 🧪 Training and testing models with train/test protocol.
- 💾 Saving trained models to a dictionary for later analysis.
- 📊 Plotting the results of a single model.
- 📈 Plotting the results of multiple models as a bar chart.
- 💾 Saving the results of multiple models as a CSV file.

"""

import importlib

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os

from docs.examples import generate_sklearn_simulated_data
from fusilli.data import get_data_module
from fusilli.eval import RealsVsPreds, ModelComparison
from fusilli.train import train_and_save_models
from fusilli.utils.model_chooser import import_chosen_fusion_models

# %%
# 1. Import fusion models 🔍
# --------------------------------
# Let's kick things off by importing our fusion models. The models are imported using the
# :func:`~fusilli.utils.model_chooser.import_chosen_fusion_models` function, which takes a dictionary of conditions
# as an input. The conditions are the attributes of the models, e.g. the class name, the modality type, etc.
#
# The function returns list of class objects that match the conditions. If no conditions are specified, then all the models are returned.
#
# We're importing ConcatTabularData and TabularChannelWiseMultiAttention models for this example. Both are multimodal tabular models.

model_conditions = {
    "class_name": ["ConcatTabularData", "TabularChannelWiseMultiAttention"],
}

fusion_models = import_chosen_fusion_models(model_conditions)

# %%
# 2. Set the training parameters 🎯
# -----------------------------------
# Now, let's configure our training parameters. The parameters are stored in a dictionary and passed to most
# of the methods in this library.
# For training and testing, the necessary parameters are:
#
# - ``test_size``: the proportion of the data to be used for testing.
# - ``kfold_flag``: the user sets this to False for train/test protocol.
# - ``log``: a boolean of whether to log the results using Weights and Biases (True) or not (False).
# - ``pred_type``: the type of prediction to be performed. This is either ``regression``, ``binary``, or ``classification``. For this example we're using regression.
# - ``loss_log_dir``: the directory to save the loss logs to. This is used for plotting the loss curves.

params = {
    "test_size": 0.2,
    "kfold_flag": False,
    "log": False,
    "pred_type": "regression",
    "loss_log_dir": "loss_logs/two_models_traintest",  # where the csv of the loss is saved for plotting later
}

# empty the loss log directory
for dir in os.listdir(params["loss_log_dir"]):
    for file in os.listdir(os.path.join(params["loss_log_dir"], dir)):
        os.remove(os.path.join(params["loss_log_dir"], dir, file))
    # remove dir
    os.rmdir(os.path.join(params["loss_log_dir"], dir))

# %%
# 3. Generating simulated data 🔮
# --------------------------------
# Time to create some simulated data for our models to work their wonders on.
# This function also simulated image data which we aren't using here.

params = generate_sklearn_simulated_data(
    num_samples=500,
    num_tab1_features=10,
    num_tab2_features=10,
    img_dims=(1, 100, 100),
    params=params,
)

# %%
# 4. Training the first fusion model 🏁
# --------------------------------------
# Here we train the first fusion model. We're using the ``train_and_save_models`` function to train and test the models.
# This function takes the following inputs:
#
# - ``trained_models_dict``: a dictionary to store the trained models.
# - ``data_module``: the data module containing the data.
# - ``params``: the parameters for training and testing.
# - ``fusion_model``: the fusion model to be trained.
#
# First we'll create a dictionary to store both the trained models so we can compare them later.
all_trained_models = {}  # create dictionary to store trained models

# %%
# To train the first model we need to:
#
# 1. *Choose the model*: We're using the first model in the ``fusion_models`` list we made earlier.
# 2. *Print the attributes of the model*: To check it's been initialised correctly.
# 3. *Create the datamodule*: This is done with the :func:`~fusilli.data.get_data_module` function. This function takes the initialised model and the parameters as inputs. It returns the datamodule.
# 4. *Train and test the model*: This is done with the :func:`~fusilli.train.train_and_save_models` function. This function takes the datamodule, the parameters, the fusion model, and the initialised model as inputs. It returns a list of the trained models (in this case, only one model).
# 5. *Add the trained model to the ``all_trained_models`` dictionary*: This is so we can compare the results of the two models later.

fusion_model = fusion_models[0]

print("Method name:", fusion_model.method_name)
print("Modality type:", fusion_model.modality_type)
print("Fusion type:", fusion_model.fusion_type)

# Create the data module
dm = get_data_module(fusion_model=fusion_model, params=params)

# Train and test
model_1_list = train_and_save_models(
    data_module=dm,
    params=params,
    fusion_model=fusion_model,
    enable_checkpointing=False,  # False for the example notebooks
    show_loss_plot=True,
)

# Add trained model to dictionary
all_trained_models[fusion_model.__name__] = model_1_list

# %%
# 5. Plotting the results of the first model 📊
# -----------------------------------------------
# Let's unveil the results of our first model's hard work. We're using the :class:`~fusilli.eval.RealsVsPreds` class to plot the results of the first model.
# This class takes the trained model as an input and returns a plot of the real values vs the predicted values from the final validation data (when using from_final_val_data).
# If you want to plot the results from the test data, you can use from_new_data instead. See the example notebook on plotting with new data for more detail.

reals_preds_model_1 = RealsVsPreds.from_final_val_data(model_1_list)

plt.show()

# %% [markdown]
# 6. Training the second fusion model 🏁
# ---------------------------------------
#  It's time for our second fusion model to shine! Here we train the second fusion model: TabularChannelWiseMultiAttention. We're using the same steps as before, but this time we're using the second model in the ``fusion_models`` list.


# %%
# Choose the model
fusion_model = fusion_models[1]

print("Method name:", fusion_model.method_name)
print("Modality type:", fusion_model.modality_type)
print("Fusion type:", fusion_model.fusion_type)

# Create the data module
dm = get_data_module(fusion_model=fusion_model, params=params)

# Train and test
model_2_list = train_and_save_models(
    data_module=dm,
    params=params,
    fusion_model=fusion_model,
    enable_checkpointing=False,  # False for the example notebooks
    show_loss_plot=True,
)

# Add trained model to dictionary
all_trained_models[fusion_model.__name__] = model_2_list

# %%
# 7. Plotting the results of the second model 📊
# -----------------------------------------------

reals_preds_model_2 = RealsVsPreds.from_final_val_data(model_2_list)

plt.show()

# %%
# 8. Comparing the results of the two models 📈
# ----------------------------------------------
# Let the ultimate showdown begin! We're comparing the results of our two models.
# We're using the :class:`~fusilli.eval.ModelComparison` class to compare the results of the two models.
# This class takes the trained models as an input and returns a plot of the results of the two models and a Pandas DataFrame of the metrics of the two models.

comparison_plot, metrics_dataframe = ModelComparison.from_final_val_data(
    all_trained_models
)

plt.show()

# %%
# 9. Saving the metrics of the two models 💾
# -------------------------------------------
# Time to archive our models' achievements. We're using the :class:`~fusilli.eval.ModelComparison` class to save the metrics of the two models.

metrics_dataframe
