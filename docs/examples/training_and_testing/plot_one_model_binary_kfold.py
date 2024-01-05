"""
Binary Classification: Training a K-Fold Model
======================================================

🚀 In this tutorial, we'll explore binary classification using K-fold cross validation. 
We'll show you how to train a fusion model using K-Fold cross-validation with multimodal tabular data. 
Specifically, we're using the :class:`~.TabularCrossmodalMultiheadAttention` model.


Key Features:

- 📥 Importing a model based on its path.
- 🧪 Training and testing a model with k-fold cross validation.
- 📈 Plotting the loss curves of each fold.
- 📊 Visualising the results of a single K-Fold model using the :class:`~.ConfusionMatrix` class.
"""

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os

from docs.examples import generate_sklearn_simulated_data
from fusilli.data import prepare_fusion_data
from fusilli.eval import ConfusionMatrix
from fusilli.train import train_and_save_models

# sphinx_gallery_thumbnail_number = -1

# %%
# 1. Import the fusion model 🔍
# --------------------------------
# We're importing only one model for this example, the :class:`~.TabularCrossmodalMultiheadAttention` model.
# Instead of using the :func:`~fusilli.utils.model_chooser.import_chosen_fusion_models` function, we're importing the model directly like with any other library method.


from fusilli.fusionmodels.tabularfusion.crossmodal_att import (
    TabularCrossmodalMultiheadAttention,
)

# %%
# 2. Set the training parameters 🎯
# -----------------------------------
# Now we're configuring our training parameters.
#
# For training and testing, the necessary parameters are:
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

# Binary task (predicting a binary variable - 0 or 1)
prediction_task = "binary"

# Set the batch size
batch_size = 32

# Enable k-fold cross-validation with k=5
kfold = True
num_folds = 5

# Setting output directories
output_paths = {
    "losses": "loss_logs/one_model_binary_kfold",
    "checkpoints": "checkpoints/one_model_binary_kfold",
    "figures": "figures/one_model_binary_kfold",
}

# Create the output directories if they don't exist
for path in output_paths.values():
    os.makedirs(path, exist_ok=True)

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
# 4. Training the fusion model 🏁
# --------------------------------------
# Now we're ready to train our model. We're using the :func:`~fusilli.train.train_and_save_models` function to train our model.
#
# First we need to create a data module using the :func:`~fusilli.data.prepare_fusion_data` function.
# This function takes the following parameters:
#
# - ``prediction_task``: the type of prediction to be performed.
# - ``fusion_model``: the fusion model to be trained.
# - ``data_paths``: the paths to the input data files.
# - ``output_paths``: the paths to the output directories.
#
# Then we pass the data module and the fusion model to the :func:`~fusilli.train.train_and_save_models` function.
# We're not using checkpointing for this example, so we set ``enable_checkpointing=False``. We're also setting ``show_loss_plot=True`` to plot the loss curves for each fold.


fusion_model = TabularCrossmodalMultiheadAttention

print("method_name:", fusion_model.method_name)
print("modality_type:", fusion_model.modality_type)
print("fusion_type:", fusion_model.fusion_type)

dm = prepare_fusion_data(prediction_task=prediction_task,
                         fusion_model=fusion_model,
                         data_paths=data_paths,
                         output_paths=output_paths,
                         kfold=kfold,
                         num_folds=num_folds,
                         batch_size=batch_size)

# train and test
single_model_list = train_and_save_models(
    data_module=dm,
    fusion_model=fusion_model,
    enable_checkpointing=False,  # False for the example notebooks
    show_loss_plot=True,
)

# %%
# 6. Plotting the results 📊
# ----------------------------
# Now we're ready to plot the results of our model.
# We're using the :class:`~.ConfusionMatrix` class to plot the confusion matrix.
# We're seeing each fold's confusion matrices separately on the right, and the confusion matrix created from the concatenated validation sets from each fold on the left.

confusion_matrix_fig = ConfusionMatrix.from_final_val_data(
    single_model_list
)
plt.show()
