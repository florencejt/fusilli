"""
Using External Test Data
========================================================================

Let's learn how to use external test data with Fusilli!
Some guidance can also be found in the :ref:`Data Loading <data-loading>` section of the documentation.

The extra step that we need to take is to provide the paths to the test data files to the functions that create evaluation figures: :class:`~fusilli.eval.RealsVsPreds.from_new_data`, :class:`~fusilli.eval.ConfusionMatrix.from_new_data`, :class:`~fusilli.eval.ModelComparison.from_new_data`.

.. note::

    It is not possible to use external test data with graph-based fusion models.


We'll rush through the first few steps of the training and testing process, as they are covered in more detail in the other example notebooks.

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


model_conditions = {
    "class_name": ["ConcatTabularData"],
}

fusion_models = import_chosen_fusion_models(model_conditions)

# Regression task
prediction_task = "regression"

# Set the batch size
batch_size = 48

# Setting output directories
output_paths = {
    "losses": "loss_logs/external_data",
    "checkpoints": "checkpoints/external_data",
    "figures": "figures/external_data",
}

for dir in output_paths.values():
    os.makedirs(dir, exist_ok=True)

# Clearing the loss logs directory (only for the example notebooks)
for dir in os.listdir(output_paths["losses"]):
    # remove files
    for file in os.listdir(os.path.join(output_paths["losses"], dir)):
        os.remove(os.path.join(output_paths["losses"], dir, file))
    # remove dir
    os.rmdir(os.path.join(output_paths["losses"], dir))

tabular1_path, tabular2_path = generate_sklearn_simulated_data(prediction_task,
                                                               num_samples=500,
                                                               num_tab1_features=10,
                                                               num_tab2_features=20)

external_tabular1_path, external_tabular2_path = generate_sklearn_simulated_data(prediction_task,
                                                                                 num_samples=100,
                                                                                 num_tab1_features=10,
                                                                                 num_tab2_features=20,
                                                                                 external=True)
data_paths = {
    "tabular1": tabular1_path,
    "tabular2": tabular2_path,
    "image": "",
}

external_data_paths = {
    "tabular1": external_tabular1_path,
    "tabular2": external_tabular2_path,
    "image": "",
}

fusion_model = fusion_models[0]

print("Method name:", fusion_model.method_name)
print("Modality type:", fusion_model.modality_type)
print("Fusion type:", fusion_model.fusion_type)

# Create the data module
dm = prepare_fusion_data(prediction_task=prediction_task,
                         fusion_model=fusion_model,
                         data_paths=data_paths,
                         output_paths=output_paths,
                         batch_size=batch_size, )

# train and test
trained_model = train_and_save_models(
    data_module=dm,
    fusion_model=fusion_model,
    enable_checkpointing=True,
    show_loss_plot=True,
)

# %%
# Evaluating with validation data
# -----------------------------------------------
# We'll start by evaluating the model with the validation data.

reals_preds_validation = RealsVsPreds.from_final_val_data(trained_model)
plt.show()

# %%
# Evaluating with external data
# ----------------------------------------------
# Now we'll evaluate the model with the external data.

reals_preds_external = RealsVsPreds.from_new_data(trained_model,
                                                  output_paths=output_paths,
                                                  test_data_paths=external_data_paths)
plt.show()

# %%
# Removing checkpoint files

for dir in os.listdir(output_paths["checkpoints"]):
    # remove files
    os.remove(os.path.join(output_paths["checkpoints"], dir))
