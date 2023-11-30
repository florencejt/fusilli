"""
Training multiple models in a loop: k-fold regression
====================================================================

Welcome to the "Comparing Multiple K-Fold Trained Fusion Models" tutorial! In this tutorial, we'll explore how to train and compare multiple fusion models for a regression task using k-fold cross-validation with multimodal tabular data. This tutorial is designed to help you understand and implement key features, including:

- 📥 Importing fusion models based on modality types.
- 🚲 Setting training parameters for your models
- 🔮 Generating simulated data for experimentation.
- 🧪 Training and evaluating multiple fusion models.
- 📈 Visualising the results of individual models.
- 📊 Comparing the performance of different models.

Let's dive into each of these steps in detail:

1. **Importing Fusion Models:**

   We begin by importing fusion models based on modality types. These models will be used in our regression task, and we'll explore various fusion strategies.

2. **Setting the Training Parameters:**

   To ensure consistent and controlled training, we define training parameters. These parameters include enabling k-fold cross-validation, specifying the prediction type (regression), and setting the batch size for training.

3. **Generating Simulated Data:**

   In this step, we generate synthetic data to simulate a real-world multimodal dataset. This dataset includes two tabular modalities, but it can also incorporate image data, although we won't use images in this example.

4. **Training All Fusion Models:**

   Now, we train all the selected fusion models using the generated data and the defined training parameters. We'll monitor the performance of each model during training and store the results for later analysis.

5. **Plotting Individual Model Results:**

   After training, we visualise the performance of each individual model. We create plots that show loss curves and performance metrics to help us understand how each model performed.

6. **Comparing Model Performance:**

   To gain insights into which fusion models perform best, we compare their performance using a violin chart. This chart provides a clear overview of how each model's performance metrics compare.

7. **Saving the Results:**

   Finally, we save the performance results of all the models as a structured DataFrame. This data can be further analyzed, exported to a CSV file, or used for additional experiments.

Now, let's walk through each of these steps in code and detail. Let's get started! 🌸
"""

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os

from docs.examples import generate_sklearn_simulated_data
from fusilli.data import get_data_module
from fusilli.eval import RealsVsPreds, ModelComparison
from fusilli.train import train_and_save_models
from fusilli.utils.model_chooser import import_chosen_fusion_models

# from IPython.utils import io  # for hiding the tqdm progress bar

# %%
# 1. Import fusion models 🔍
# ---------------------------
# Here we import the fusion models to be compared. The models are imported using the
# :func:`~fusilli.utils.model_chooser.get_models` function, which takes a dictionary of conditions
# as an input. The conditions are the attributes of the models, e.g. the class name, the modality type, etc.
#
# The function returns a dataframe of the models that match the conditions. The dataframe contains the
# method name, the class name, the modality type, the fusion type, the path to the model, and the path to the
# model's parent class. The paths are used to import the models with the :func:`importlib.import_module`.
#
# We're importing all the fusion models that use only tabular data for this example (either uni-modal or multi-modal).

model_conditions = {
    "modality_type": ["tabular1", "tabular2", "tabular_tabular"],
}

fusion_models = import_chosen_fusion_models(model_conditions)

# %%
# 2. Set the training parameters 🎯
# ---------------------------------
# Let's configure our training parameters. The parameters are stored in a dictionary and passed to most
# of the methods in this library.
# For training and testing, the necessary parameters are:
#
# - ``kfold_flag``: the user sets this to True for k-fold cross validation.
# - ``num_k``: the number of folds to use. It can't be k=1.
# - ``log``: a boolean of whether to log the results using Weights and Biases (True) or not (False).
# - ``pred_type``: the type of prediction to be performed. This is either ``regression``, ``binary``, or ``classification``. For this example we're using regression.
# - ``loss_log_dir``: the directory to save the loss logs to. This is used for plotting the loss curves with ``log=False``.
#
# We're also setting our own batch_size for this example.


params = {
    "kfold_flag": True,
    "num_k": 3,
    "log": False,
    "pred_type": "regression",
    "batch_size": 32,
    "loss_log_dir": "loss_logs/model_comparison_loop_kfold",
}

for dir in os.listdir(params["loss_log_dir"]):
    # remove files
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
    num_tab2_features=20,
    img_dims=(1, 100, 100),
    params=params,
)

# %%
# 4. Training the all the fusion models 🏁
# -----------------------------------------
# In this section, we train all the fusion models using the generated data and specified parameters.
# We store the results of each model for later analysis.

# Using %%capture to hide the progress bar and plots (there are a lot of them!)

all_trained_models = {}

for i, fusion_model in enumerate(fusion_models):
    fusion_model_name = fusion_model.__name__
    print(f"Running model {fusion_model_name}")

    # Get data module
    data_module = get_data_module(fusion_model, params, batch_size=params["batch_size"])

    # Train and test
    single_model_list = train_and_save_models(
        data_module=data_module,
        params=params,
        fusion_model=fusion_model,
        enable_checkpointing=False,  # False for the example notebooks
        show_loss_plot=True,  # True for the example notebooks
    )

    # Save to all_trained_models
    all_trained_models[fusion_model_name] = single_model_list

    plt.close("all")

# %%
# 5. Plotting the results of the individual models
# -------------------------------------------------
# In this section, we visualize the results of each individual model.
#
# If you want to save the figures rather than show them, you can use the :meth:`~.save_to_local' method of the :class:`~fusilli.eval.Plotter` class.
# This will save the figures in a timestamped folder in the current working directory with the method name and plot type in the filename.
# You can add an extra suffix to the filename by passing a string to the ``extra_string`` argument of the :meth:`~fusilli.eval.Plotter.save_to_local` method.

for model_name, model_list in all_trained_models.items():
    fig = RealsVsPreds.from_final_val_data(model_list)
    plt.show()

# %% [markdown]
# 6. Plotting comparison of the models
# -------------------------------------
# In this section, we visualize the results of each individual model.

comparison_plot, metrics_dataframe = ModelComparison.from_final_val_data(all_trained_models)
plt.show()

# %%
# 7. Saving the results of the models
# -------------------------------------
# In this section, we compare the performance of all the trained models using a violin chart, providing an overview of how each model performed as a distribution over the different cross-validation folds.


metrics_dataframe
