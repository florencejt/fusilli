"""
Comparing Multiple K-Fold Trained Fusion Models
====================================================================

Welcome to the "Comparing Multiple K-Fold Trained Fusion Models" tutorial! In this tutorial, we'll explore how to train and compare multiple fusion models for a regression task using k-fold cross-validation with multimodal tabular data. This tutorial is designed to help you understand and implement key features, including:

1. Importing fusion models based on modality types.
2. Setting training parameters for your models.
3. Generating simulated data for experimentation.
4. Training and evaluating multiple fusion models.
5. Visualizing the results of individual models.
6. Comparing the performance of different models.
7. Saving the results for further analysis.

Let's dive into each of these steps in detail:

1. **Importing Fusion Models:**

   We begin by importing fusion models based on modality types. These models will be used in our regression task, and we'll explore various fusion strategies. The imported models will provide flexibility in model selection.

2. **Setting the Training Parameters:**

   To ensure consistent and controlled training, we define training parameters. These parameters include enabling k-fold cross-validation, specifying the prediction type (regression), and setting the batch size for training.

3. **Generating Simulated Data:**

   In this step, we generate synthetic data to simulate a real-world multimodal dataset. This dataset includes two tabular modalities, but it can also incorporate image data, although we won't use images in this example.

4. **Training All Fusion Models:**

   Now, we train all the selected fusion models using the generated data and the defined training parameters. We'll monitor the performance of each model during training and store the results for later analysis.

5. **Plotting Individual Model Results:**

   After training, we visualize the performance of each individual model. We create plots that show loss curves and performance metrics to help us understand how each model performed.

6. **Comparing Model Performance:**

   To gain insights into which fusion models perform best, we compare their performance using a violin chart. This chart provides a clear overview of how each model's performance metrics compare.

7. **Saving the Results:**

   Finally, we save the performance results of all the models as a structured DataFrame. This data can be further analyzed, exported to a CSV file, or used for additional experiments.

Now, let's walk through each of these steps in code and detail. Let's get started!
"""

import importlib

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from docs.examples import generate_sklearn_simulated_data
from fusionlibrary.data import get_data_module
from fusionlibrary.eval import Plotter
from fusionlibrary.train import train_and_save_models
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
# We're importing all the fusion models that use only tabular data for this example (either uni-modal or multi-modal).

model_conditions = {
    "modality_type": ["tabular1", "tabular2", "both_tab"],
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
# In this section, we set the training parameters for our models. These parameters include k-fold cross-validation, prediction type (regression), and batch size.


params = {
    "kfold_flag": True,
    "num_k": 10,
    "log": False,
    "pred_type": "regression",
    "batch_size": 32,
}


# %%
# 3. Generating simulated data
# --------------------------------
# Here we generate simulated data for the two tabular modalities for this example.
# This function also simulated image data which we aren't using here.

params = generate_sklearn_simulated_data(
    num_samples=500,
    num_tab1_features=10,
    num_tab2_features=20,
    img_dims=(1, 100, 100),
    params=params,
)

# %%
# 4. Training the all the fusion models
# ---------------------------------------
# In this section, we train all the fusion models using the generated data and specified parameters.
# We store the results of each model for later analysis.

all_trained_models = {}
single_model_dicts = []  # for plotting single models later

for i, fusion_model in enumerate(fusion_models):
    print(f"Running model {fusion_model.__name__}")

    # Get data module
    data_module = get_data_module(fusion_model, params, batch_size=params["batch_size"])

    # Train and test
    single_model_dict = train_and_save_models(
        data_module=data_module,
        params=params,
        fusion_model=fusion_model,
        enable_checkpointing=False,  # False for the example notebooks
    )

    # Save to all_trained_models
    all_trained_models[fusion_model.__name__] = single_model_dict[fusion_model.__name__]
    single_model_dicts.append(single_model_dict)

# %%
# 5. Plotting the results of the individual models
# -------------------------------------------------
# In this section, we visualize the results of each individual model.
#
# If you want to save the figures rather than show them, you can use the :meth:`~.save_to_local' method of the :class:`~fusionlibrary.eval.Plotter` class.
# This will save the figures in a timestamped folder in the current working directory with the method name and plot type in the filename.
# You can add an extra suffix to the filename by passing a string to the ``extra_string`` argument of the :meth:`~fusionlibrary.eval.Plotter.save_to_local` method.

for model_dict in single_model_dicts:
    plotter = Plotter(model_dict, params)
    single_model_figures_dict = plotter.plot_all()
    plotter.show_all(single_model_figures_dict)

# %% [markdown]
# 6. Plotting comparison of the models
# -------------------------------------
# In this section, we visualize the results of each individual model.

comparison_plotter = Plotter(all_trained_models, params)
comparison_plot_dict = comparison_plotter.plot_all()
comparison_plotter.show_all(comparison_plot_dict)

# %%
# 7. Saving the results of the models
# -------------------------------------
# In this section, we compare the performance of all the trained models using a violin chart, providing an overview of how each model performed as a distribution over the different cross-validation folds.


performances_df = comparison_plotter.get_performance_df()
performances_df
