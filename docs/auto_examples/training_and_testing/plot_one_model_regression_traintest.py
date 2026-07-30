"""
Train/Test split: Regression
======================================================

🚀 In this tutorial, we'll explore regression using a train/test split.
Specifically, we're using the :class:`~.TabularCrossmodalMultiheadAttention` model.

Data:

The data we are using is 500 rows of the MNIST dataset, split into top and bottom halves as our two tabular modalities.
The bottom half's values have been inverted to make the task more difficult.
The prediction labels (the number shown in the image) has been changed into a continuous variable (1.0, 2.0, 3.0, etc.) and had some noise added to it.
So the labels look more like 1.05, 2.02, 3.01, etc.

Key Features:

- 📥 Importing a model based on its path.
- 🧪 Training and testing a model with train/test split.
- 📈 Plotting the loss curves of each fold.
- 📊 Visualising the results of a single train/test model using the :class:`~.RealsVsPreds` class.
"""

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os

from fusilli.data import prepare_fusion_data
from fusilli.eval import RealsVsPreds
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

# Regression task
prediction_task = "regression"

# Set the batch size
batch_size = 32

# Setting output directories
output_paths = {
    "losses": "loss_logs/one_model_regression_traintest",
    "checkpoints": "checkpoints/one_model_regression_traintest",
    "figures": "figures/one_model_regression_traintest",
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
# 3. Specifying input file paths 🔮
# --------------------------------
# We're using the MNIST dataset for this example, and the CSV files are stored in the ``_static/mnist_data`` directory with the documentation files.


data_paths = {
    "tabular1": "../../_static/mnist_data/mnist1_regression.csv",
    "tabular2": "../../_static/mnist_data/mnist2_regression.csv",
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
# We're not using checkpointing for this example, so we set ``enable_checkpointing=False``. We're also setting ``show_loss_plot=True`` to plot the loss curve.


fusion_model = TabularCrossmodalMultiheadAttention

print("method_name:", fusion_model.method_name)
print("modality_type:", fusion_model.modality_type)
print("fusion_type:", fusion_model.fusion_type)

dm = prepare_fusion_data(prediction_task=prediction_task,
                         fusion_model=fusion_model,
                         data_paths=data_paths,
                         output_paths=output_paths,
                         batch_size=batch_size)

# train and test
single_model_list = train_and_save_models(
    data_module=dm,
    fusion_model=fusion_model,
    enable_checkpointing=False,  # False for the example notebooks
    show_loss_plot=True,
    metrics_list=["r2", "mae", "mse"]
)

# %%
# 6. Plotting the results 📊
# ----------------------------
# Now we're ready to plot the results of our model.
# We're using the :class:`~.RealsVsPreds` class to plot the confusion matrix.

reals_preds_fig = RealsVsPreds.from_final_val_data(
    single_model_list
)
plt.show()
