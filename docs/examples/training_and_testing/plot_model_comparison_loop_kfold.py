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
from fusilli.data import prepare_fusion_data
from fusilli.eval import RealsVsPreds, ModelComparison
from fusilli.train import train_and_save_models
from fusilli.utils.model_chooser import import_chosen_fusion_models

# sphinx_gallery_thumbnail_number = -1

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
# Let's configure our training parameters.
# For training and testing, the necessary parameters are:
#
# - Paths to the input data files.
# - Paths to the output directories.
# - ``prediction_task``: the type of prediction to be performed. This is either ``regression``, ``binary``, or ``classification``.
#
# Some optional parameters are:
#
# - ``kfold``: a boolean of whether to use k-fold cross-validation (True) or not (False). By default, this is set to False.
# - ``num_folds``: the number of folds to use. It can't be ``k=1``.
# - ``wandb_logging``: a boolean of whether to log the results using Weights and Biases (True) or not (False). Default is False.
# - ``test_size``: the proportion of the dataset to include in the test split. Default is 0.2.
# - ``batch_size``: the batch size to use for training. Default is 8.
# - ``multiclass_dimensions``: the number of classes to use for multiclass classification. Default is None unless ``prediction_task`` is ``multiclass``.
# - ``max_epochs``: the maximum number of epochs to train for. Default is 1000.

# Regression task (predicting a continuous variable)
prediction_task = "regression"

# Set the batch size
batch_size = 32

# Enable k-fold cross-validation with k=3
kfold = True
num_folds = 3

# Setting output directories
output_paths = {
    "losses": "loss_logs/model_comparison_loop_kfold",
    "checkpoints": "checkpoints/model_comparison_loop_kfold",
    "figures": "figures/model_comparison_loop_kfold",
}

os.makedirs(output_paths["losses"], exist_ok=True)
os.makedirs(output_paths["checkpoints"], exist_ok=True)
os.makedirs(output_paths["figures"], exist_ok=True)

# Clearing the loss logs directory (only for the example notebooks)
for dir in os.listdir(output_paths["losses"]):
    # remove files
    for file in os.listdir(os.path.join(output_paths["losses"], dir)):
        os.remove(os.path.join(output_paths["losses"], dir, file))
    # remove dir
    os.rmdir(os.path.join(output_paths["losses"], dir))

# %%
# 3. Generating simulated data 🔮
# --------------------------------
# Time to create some simulated data for our models to work their wonders on.
# This function also simulated image data which we aren't using here.

tabular1_path, tabular2_path = generate_sklearn_simulated_data(prediction_task,
                                                               num_samples=500,
                                                               num_tab1_features=10,
                                                               num_tab2_features=20)

data_paths = {
    "tabular1": tabular1_path,
    "tabular2": tabular2_path,
    "image": "",
}

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
    data_module = prepare_fusion_data(prediction_task=prediction_task,
                                      fusion_model=fusion_model,
                                      data_paths=data_paths,
                                      output_paths=output_paths,
                                      kfold=kfold,
                                      num_folds=num_folds,
                                      batch_size=batch_size)

    # Train and test
    single_model_list = train_and_save_models(
        data_module=data_module,
        fusion_model=fusion_model,
        enable_checkpointing=False,  # We're not saving the trained models for this example
        show_loss_plot=True,  # We'll show the loss plot for each model instead of saving it
    )

    # Save to all_trained_models
    all_trained_models[fusion_model_name] = single_model_list

    plt.close("all")

# %%
# 5. Plotting the results of the individual models
# -------------------------------------------------
# In this section, we visualize the results of each individual model.
#

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
