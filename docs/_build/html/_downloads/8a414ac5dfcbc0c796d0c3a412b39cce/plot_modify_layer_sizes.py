"""
How to modify architectures of fusion models
############################################

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
# For a more detailed explanation of this process, please see the :ref:`train_test_examples` tutorials.
#

import matplotlib.pyplot as plt
import os
import torch.nn as nn

from docs.examples import generate_sklearn_simulated_data
from fusilli.data import get_data_module
from fusilli.eval import RealsVsPreds
from fusilli.train import train_and_save_models

from fusilli.fusionmodels.tabularimagefusion.denoise_tab_img_maps import DAETabImgMaps

params = {
    "test_size": 0.2,
    "kfold_flag": False,
    "log": False,
    "pred_type": "regression",
    "loss_log_dir": "loss_logs/modify_layers",  # where the csv of the loss is saved for plotting later
    "checkpoint_dir": "checkpoints",
    "loss_fig_path": "loss_figures",
}

# empty the loss log directory
for dir in os.listdir(params["loss_log_dir"]):
    for file in os.listdir(os.path.join(params["loss_log_dir"], dir)):
        os.remove(os.path.join(params["loss_log_dir"], dir, file))
    # remove dir
    os.rmdir(os.path.join(params["loss_log_dir"], dir))

params = generate_sklearn_simulated_data(
    num_samples=500,
    num_tab1_features=10,
    num_tab2_features=10,
    img_dims=(1, 100, 100),
    params=params,
)

# %%
# Specifying the model modifications
# ----------------------------------
#
# Now, we will specify the modifications we want to make to the model.
#
# We are using the :class:`~fusilli.fusionmodels.tabularimagefusion.denoise_tab_img_maps.DAETabImgMaps` model for this example.
# This is a subspace-based model which has two PyTorch models that need to be pretrained (a denoising autoencoder for the tabular modality, and a convolutional neural network for the image modality).
# The fusion model then uses the latent representations of these models to perform the fusion.
#
# The following modifications can be made to the **pre-trained subspace** model :class:`~fusilli.fusionmodels.tabularimagefusion.denoise_tab_img_maps.denoising_autoencoder_subspace_method`:
#
# .. list-table::
#   :widths: 40 60
#   :header-rows: 1
#   :stub-columns: 0
#
#   * - Attribute
#     - Guidance
#   * - :attr:`.autoencoder.latent_dim`
#     - int
#   * - :attr:`.autoencoder.upsampler`
#     - ``nn.Sequential``
#   * - :attr:`.autoencoder.downsampler`
#     - ``nn.Sequential``
#   * - :attr:`.img_unimodal.img_layers`
#     -
#       * ``nn.Sequential``
#       * Overrides modification of ``img_layers`` made to "all"
#   * - :attr:`.img_unimodal.fused_layers`
#     - ``nn.Sequential``
#
# The following modifications can be made to the **fusion** model :class:`~fusilli.fusionmodels.tabularimagefusion.denoise_tab_img_maps.DAETabImgMaps`:
#
# .. list-table::
#   :widths: 40 60
#   :header-rows: 1
#   :stub-columns: 0
#
#   * - Attribute
#     - Guidance
#   * - :attr:`~.DAETabImgMaps.fusion_layers`
#     - ``nn.Sequential``
#
# Let's change everything that we can!

layer_mods = {
    "DAETabImgMaps": {
        "fusion_layers": nn.Sequential(
            nn.Linear(20, 420),
            nn.ReLU(),
            nn.Linear(420, 100),
            nn.ReLU(),
            nn.Linear(100, 78),
        ),
    },
    "denoising_autoencoder_subspace_method": {
        "autoencoder.latent_dim": 150,  # denoising autoencoder latent dim
        "autoencoder.upsampler": nn.Sequential(
            nn.Linear(20, 80),
            nn.ReLU(),
            nn.Linear(80, 100),
            nn.ReLU(),
            nn.Linear(100, 150),
            nn.ReLU(),
        ),
        "autoencoder.downsampler": nn.Sequential(
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, 80),
            nn.ReLU(),
            nn.Linear(80, 20),
            nn.ReLU(),
        ),
        "img_unimodal.img_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Conv2d(1, 40, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
                "layer 2": nn.Sequential(
                    nn.Conv2d(40, 60, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
                "layer 3": nn.Sequential(
                    nn.Conv2d(60, 85, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
            }
        ),
        "img_unimodal.fused_layers": nn.Sequential(
            nn.Linear(85, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
    },
}

# %%
# Loading the data and training the model
# ---------------------------------------


# load data
datamodule = get_data_module(DAETabImgMaps, params, layer_mods=layer_mods, max_epochs=5, batch_size=64)

# train
trained_model_list = train_and_save_models(
    data_module=datamodule,
    params=params,
    fusion_model=DAETabImgMaps,
    layer_mods=layer_mods,
    max_epochs=5,
)

# %%
# It worked! Let's have a look at the model structure to see what changes have been made.

print("Subspace Denoising Autoencoder:\n", datamodule.subspace_method_train.autoencoder)
print("Subspace Image CNN:\n", datamodule.subspace_method_train.img_unimodal)
print("Fusion model:\n", trained_model_list[0].model)

# %%
# What happens when the modifications are incorrect?
# ----------------------------------------------------
#
# Let's see what happens when we try to modify an **attribute that doesn't exist**.
#

layer_mods = {
    "denoising_autoencoder_subspace_method": {
        "autoencoder.fake_layers": nn.Sequential(
            nn.Linear(20, 420),
            nn.Linear(420, 100),
            nn.Linear(100, 78),
        ),
    }
}

try:
    datamodule = get_data_module(DAETabImgMaps, params, layer_mods=layer_mods, max_epochs=5, batch_size=64)
except Exception as error:
    print(error)

# %%
# What about modifying an attribute with the **wrong data type**?
#
# * ``latent_dim`` should be an ``int`` and greater than 0.
# * ``upsampler`` should be an ``nn.Sequential``
# * ``downsampler`` should be an ``nn.Sequential``
# * ``img_layers`` should be an ``nn.ModuleDict``

layer_mods = {
    "denoising_autoencoder_subspace_method": {
        "autoencoder.latent_dim": 0,
    }
}

try:
    get_data_module(DAETabImgMaps, params, layer_mods=layer_mods, max_epochs=5, batch_size=64)
except Exception as error:
    print(error)

# %%

layer_mods = {
    "denoising_autoencoder_subspace_method": {
        "autoencoder.upsampler": nn.Linear(10, 10),
    }
}

try:
    get_data_module(DAETabImgMaps, params, layer_mods=layer_mods, max_epochs=5, batch_size=64)
except Exception as error:
    print(error)

# %%
# What about modifying multiple attributes with the **conflicting modifications**?
#
# For this, let's modify the ``latent_dim`` and the ``upsampler``. of the ``autoencoder`` model.
# The output of the ``upsampler`` should be the same size as the ``latent_dim``.
# If we modify both of these to be mismatched, let's see what happens.

layer_mods = {
    "denoising_autoencoder_subspace_method": {
        "autoencoder.latent_dim": 450,
        "autoencoder.upsampler": nn.Sequential(
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 300),  # this should be 450 to match the latent_dim
            nn.ReLU(),
        )
    },
}

# get the data and train the subspace models
datamodule = get_data_module(DAETabImgMaps, params, layer_mods=layer_mods, max_epochs=5, batch_size=64)

# %%
# **Wow it still works!**
# Let's have a look at what the model structure looks like to see what changes have been made to keep the model valid.

print(datamodule.subspace_method_train.autoencoder)

# %%
# As you can see, a few corrections have been made to the modifications:
#
# * The ``upsampler`` has been modified to have the correct number of nodes in the final layer to match the ``latent_dim``.
# * The ``downsample`` (which we didn't specify a modification for) now has the correct number of nodes in the first layer to match the ``latent_dim``.
#
# In general, there are checks in the fusion models to make sure that the modifications are valid.
# If the input number of nodes to a modification is not correct, then the model will automatically calculate the correct number of nodes and correct the modification.
#
# This is the case for quite a few modifications, but potentially not all of them so please be careful!
# Make sure to print out the model structure to check that the modifications have been made correctly and see what changes have been made to keep the model valid.

# removing checkpoints
os.remove(params["checkpoint_dir"] + "/DAETabImgMaps_epoch=04.ckpt")
os.remove(params["checkpoint_dir"] + "/subspace_DAETabImgMaps_DenoisingAutoencoder.ckpt")
os.remove(params["checkpoint_dir"] + "/subspace_DAETabImgMaps_ImgUnimodalDAE.ckpt")
