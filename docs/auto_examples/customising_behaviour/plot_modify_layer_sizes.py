"""
How to modify fusion model architecture
################################################

This tutorial will show you how to modify the architectures of fusion models.

More guidance on what can be modified in each fusion model can be found in the :ref:`modifying-models` section.

.. warning::

    Some of the fusion models have been designed to work with specific architectures and there are some restrictions on how they can be modified.

    For example, the channel-wise attention model requires the two modalities to have the same number of layers. Please read the notes section of the fusion model you are interested in to see if there are any restrictions.

"""

# %%
# Setting up the experiment
# -------------------------
#
# First, we will set up the experiment by importing the necessary packages, creating the simulated data, and setting the parameters for the experiment.
#
# For a more detailed explanation of this process, please see the example tutorials.
#

# sphinx_gallery_thumbnail_path = '_static/modify_thumbnail.png'
import matplotlib.pyplot as plt
import os
import torch.nn as nn
from torch_geometric.nn import GCNConv, ChebConv

from docs.examples import generate_sklearn_simulated_data
from fusilli.data import prepare_fusion_data
from fusilli.eval import RealsVsPreds
from fusilli.train import train_and_save_models

from fusilli.fusionmodels.tabularfusion.attention_weighted_GNN import AttentionWeightedGNN

prediction_task = "regression"

output_paths = {
    "checkpoints": "checkpoints",
    "losses": "loss_logs/modify_layers",
    "figures": "loss_figures",
}

for dir in output_paths.values():
    os.makedirs(dir, exist_ok=True)

# empty the loss log directory (only needed for this tutorial)
for dir in os.listdir(output_paths["losses"]):
    for file in os.listdir(os.path.join(output_paths["losses"], dir)):
        os.remove(os.path.join(output_paths["losses"], dir, file))
    # remove dir
    os.rmdir(os.path.join(output_paths["losses"], dir))

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
# Specifying the model modifications
# ----------------------------------
#
# Now, we will specify the modifications we want to make to the model.
#
# We are using the :class:`~fusilli.fusionmodels.tabularfusion.attention_weighted_GNN.AttentionWeightedGNN` model for this example.
# This is a graph-based model which has a pretrained MLP (multi-layer perceptron) to get attention weights, and a graph neural network that uses the attention weights to perform the fusion.
#
# The following modifications can be made to the method that makes the graph structure: :class:`~fusilli.fusionmodels.tabularfusion.attention_weighted_GNN.AttentionWeightedGraphMaker`:
#
#
# .. list-table::
#   :widths: 40 60
#   :header-rows: 1
#   :stub-columns: 0
#
#   * - Attribute
#     - Guidance
#   * - :attr:`~.AttentionWeightedGraphMaker.early_stop_callback`
#     - ``EarlyStopping`` object from ``from lightning.pytorch.callbacks import EarlyStopping``
#   * - :attr:`~.AttentionWeightedGraphMaker.edge_probability_threshold`
#     - Integer between 0 and 100.
#   * - :attr:`~.AttentionWeightedGraphMaker.attention_MLP_test_size`
#     - Float between 0 and 1.
#   * - :attr:`~.AttentionWeightedGraphMaker.AttentionWeightingMLPInstance.weighting_layers`
#     - ``nn.ModuleDict``: final layer output size must be the same as the input layer input size.
#   * - :attr:`~.AttentionWeightedGraphMaker.AttentionWeightingMLPInstance.fused_layers`
#     - ``nn.Sequential``
#
#
# The following modifications can be made to the **fusion** model :class:`~fusilli.fusionmodels.tabularfusion.attention_weighted_GNN.AttentionWeightedGNN`:
#
# .. list-table::
#   :widths: 40 60
#   :header-rows: 1
#   :stub-columns: 0
#
#   * - Attribute
#     - Guidance
#   * - :attr:`~.AttentionWeightedGNN.graph_conv_layers`
#     - ``nn.Sequential`` of ``torch_geometric.nn`` Layers.
#   * - :attr:`~.AttentionWeightedGNN.dropout_prob`
#     - Float between (not including) 0 and 1.
#
# Let's modify the model! More info about how to do this can be found in :ref:`modifying-models`.

layer_mods = {
    "AttentionWeightedGNN": {
        "graph_conv_layers": nn.Sequential(
            ChebConv(15, 50, K=3),
            ChebConv(50, 100, K=3),
            ChebConv(100, 130, K=3),
        ),
        "dropout_prob": 0.4,
    },
    "AttentionWeightedGraphMaker": {
        "edge_probability_threshold": 80,
        "attention_MLP_test_size": 0.3,
        "AttentionWeightingMLPInstance.weighting_layers": nn.ModuleDict(
            {
                "Layer 1": nn.Sequential(
                    nn.Linear(25, 100),
                    nn.ReLU()),
                "Layer 2": nn.Sequential(
                    nn.Linear(100, 75),
                    nn.ReLU()),
                "Layer 3": nn.Sequential(
                    nn.Linear(75, 75),
                    nn.ReLU()),
                "Layer 4": nn.Sequential(
                    nn.Linear(75, 100),
                    nn.ReLU()),
                "Layer 5": nn.Sequential(
                    nn.Linear(100, 30),
                    nn.ReLU()),
            }
        )},
}

# %%
# Loading the data and training the model
# ---------------------------------------


# load data
datamodule = prepare_fusion_data(prediction_task=prediction_task,
                                 fusion_model=AttentionWeightedGNN,
                                 data_paths=data_paths,
                                 output_paths=output_paths,
                                 layer_mods=layer_mods,
                                 max_epochs=5)

# train
trained_model_list = train_and_save_models(
    data_module=datamodule,
    fusion_model=AttentionWeightedGNN,
    layer_mods=layer_mods,
    max_epochs=5,
)

# %%
# It worked! Let's have a look at the model structure to see what changes have been made.

print("Attention Weighted MLP:\n", datamodule.graph_maker_instance.AttentionWeightingMLPInstance)
print("Fusion model:\n", trained_model_list[0].model)

# %%
# You can see that the input features to the ``final_prediction`` layer changed to fit with our modification to the ``graph_conv_layers`` output features!
#
# What happens when the modifications are incorrect?
# ----------------------------------------------------
#
# Let's see what happens when we try to modify an **attribute that doesn't exist**.
#

layer_mods = {
    "AttentionWeightedGraphMaker": {
        "AttentionWeightingMLPInstance.fake_attribute": nn.Sequential(
            nn.Linear(25, 100),
            nn.ReLU(),
        ),
    }
}

try:
    datamodule = prepare_fusion_data(prediction_task=prediction_task,
                                     fusion_model=AttentionWeightedGNN,
                                     data_paths=data_paths,
                                     output_paths=output_paths,
                                     layer_mods=layer_mods,
                                     max_epochs=5)
except Exception as error:
    print(error)

# %%
# What about modifying an attribute with the **wrong data type**?
#
# * ``dropout_prob`` should be an ``float`` and between 0 and 1.
# * ``graph_conv_layers`` should be an ``nn.Sequential`` of graph convolutional layers.
# * ``edge_probability_threshold`` should be a ``float`` between 0 and 100.
# * ``AttentionWeightingMLPInstance.weighting_layers`` should be an ``nn.ModuleDict``

layer_mods = {
    "AttentionWeightedGraphMaker": {
        "AttentionWeightingMLPInstance.weighting_layers": nn.Sequential(
            nn.Linear(25, 75),
            nn.ReLU(),
            nn.Linear(75, 75),
            nn.ReLU(),
            nn.Linear(75, 25),
            nn.ReLU()
        ),
    }
}

try:
    prepare_fusion_data(prediction_task=prediction_task,
                        fusion_model=AttentionWeightedGNN,
                        data_paths=data_paths,
                        output_paths=output_paths,
                        layer_mods=layer_mods,
                        max_epochs=5)
except Exception as error:
    print(error)

# %%

layer_mods = {
    "AttentionWeightedGraphMaker": {
        "edge_probability_threshold": "two",
    }
}

try:
    prepare_fusion_data(prediction_task=prediction_task,
                        fusion_model=AttentionWeightedGNN,
                        data_paths=data_paths,
                        output_paths=output_paths,
                        layer_mods=layer_mods,
                        max_epochs=5)
except Exception as error:
    print(error)

# %%
# What about modifying multiple attributes with the **conflicting modifications**?
# -------------------------------------------------------------------------------------
#
#
# For this, let's switch to looking at the :class:`~fusilli.fusionmodels.tabularfusion.concat_feature_maps.ConcatTabularFeatureMaps` model.
# This model concatenates the feature maps of the two modalities and then passes them through a prediction layer.
#
# We can modify the layers that each tabular modality goes through before being concatenated, as well as the layers that come after the concatenation.
#
# The output features of our modified ``mod1_layers`` and ``mod2_layers`` are 100 and 128, so the input features of the ``fused_layers`` should be 228. However, we've set the input features of the ``fused_layers`` to be 25.
#
# Let's see what happens when we try to modify the model in this way. It should throw an error when the data is passed through the model.

layer_mods = {
    "ConcatTabularFeatureMaps": {
        "mod1_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(10, 32),
                    nn.ReLU(),
                ),
                "layer 2": nn.Sequential(
                    nn.Linear(32, 66),
                    nn.ReLU(),
                ),
                "layer 3": nn.Sequential(
                    nn.Linear(66, 128),
                    nn.ReLU(),
                ),
            }
        ),
        "mod2_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(15, 45),
                    nn.ReLU(),
                ),
                "layer 2": nn.Sequential(
                    nn.Linear(45, 70),
                    nn.ReLU(),
                ),
                "layer 3": nn.Sequential(
                    nn.Linear(70, 100),
                    nn.ReLU(),
                ),
            }
        ),
        "fused_layers": nn.Sequential(
            nn.Linear(25, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
    },
}

# get the data and train the model

from fusilli.fusionmodels.tabularfusion.concat_feature_maps import ConcatTabularFeatureMaps

datamodule = prepare_fusion_data(prediction_task=prediction_task,
                                 fusion_model=ConcatTabularFeatureMaps,
                                 data_paths=data_paths,
                                 output_paths=output_paths,
                                 layer_mods=layer_mods,
                                 max_epochs=5)
trained_model_list = train_and_save_models(
    data_module=datamodule,
    fusion_model=ConcatTabularFeatureMaps,
    layer_mods=layer_mods,
    max_epochs=5,
)

# %%
# **Wow it still works!**
# Let's have a look at what the model structure looks like to see what changes have been made to keep the model valid.

print(trained_model_list[0].model)

# %%
# As you can see, a few corrections have been made to the modifications:
#
# * The ``fused_layers`` has been modified to have the correct number of nodes in the first layer to match the concatenated feature maps from the two modalities.
#
# In general, there are checks in the fusion models to make sure that the modifications are valid.
# If the input number of nodes to a modification is not correct, then the model will automatically calculate the correct number of nodes and correct the modification.
#
# This is the case for quite a few modifications, but potentially not all of them so please be careful!
# Make sure to print out the model structure to check that the modifications have been made correctly and see what changes have been made to keep the model valid.

# removing checkpoints

for file in os.listdir(output_paths["checkpoints"]):
    # remove file
    os.remove(os.path.join(output_paths["checkpoints"], file))
