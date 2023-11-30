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
from fusilli.data import get_data_module
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
# For using k-fold cross validation, the necessary parameters are:
#
# - ``kfold_flag``: the user sets this to True for k-fold cross validation.
# - ``num_k``: the number of folds to use. It can't be k=1.
# - ``log``: a boolean of whether to log the results using Weights and Biases (True) or not (False).
# - ``pred_type``: the type of prediction to be performed. This is either ``regression``, ``binary``, or ``classification``. For this example we're using binary classification.
# - ``loss_log_dir``: the directory to save the loss logs to. This is used for plotting the loss curves with ``log=False``.
#
# We're also setting our own batch_size for this example.

params = {
    "kfold_flag": True,
    "num_k": 5,  # number of folds
    "log": False,
    "pred_type": "binary",
    "batch_size": 32,
    "loss_log_dir": "loss_logs/one_model_binary_kfold",
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
# 4. Training the fusion model 🏁
# --------------------------------------
# Now we're ready to train our model. We're using the :func:`~fusilli.train.train_and_save_models` function to train our model.
#
# First we need to create a data module using the :func:`~fusilli.data.get_data_module` function.
# This function takes the following parameters:
#
# - ``fusion_model``: the fusion model to be trained.
# - ``params``: the parameters for training and testing.
# - ``batch_size``: the batch size for training and testing. This is optional and defaults to 8.
#
# Then we pass the data module, the parameters, and the fusion model to the :func:`~fusilli.train.train_and_save_models` function.
# We're not using checkpointing for this example, so we set ``enable_checkpointing=False``. We're also setting ``show_loss_plot=True`` to plot the loss curves for each fold.


fusion_model = TabularCrossmodalMultiheadAttention

print("method_name:", fusion_model.method_name)
print("modality_type:", fusion_model.modality_type)
print("fusion_type:", fusion_model.fusion_type)

dm = get_data_module(
    fusion_model=fusion_model, params=params, batch_size=params["batch_size"]
)

# train and test
single_model_list = train_and_save_models(
    data_module=dm,
    params=params,
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
