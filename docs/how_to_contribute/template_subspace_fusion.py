"""
Creating a subspace-based fusion model
=======================================

This tutorial will show you how to create a subspace-based fusion model.

.. note::

    I recommend looking at :ref:`how_to_contribute_a_template_other_fusion` before looking at this template, as I will skip over some of the details that are covered in that template (particularly regarding documentation and idiosyncrasies of the fusion model template).

There are **two** types of subspace-based fusion models in this library:

1. A model that has subspace methods trained **before** the main prediction model. An example of this is :class:`~fusilli.fusionmodels.tabularfusion.denoise_tab_img_maps.DAETabImgMaps`. This works by training the subspace method first, then using the output of the subspace method as the input to the main prediction model.
2. A model that has subspace methods (such as an autoencoder) trained **simultaneously** with the main prediction model. An example of this is :class:`~fusilli.fusionmodels.tabularimagefusion.concat_img_latent_tab_doubleloss.ConcatImgLatentTabDoubleLoss`. This works by implementing a joint loss function that combines the loss of the subspace method and the loss of the main prediction model.

We will look at how to create both of these types of models.
"""

# %%
# Option 1: how to create a pre-trained subspace-based fusion model
# ------------------------------------------------------------------
#
# * Same as a general fusion model except a couple differences
# * Must have the attribute ``self.custom_loss`` which is the loss function used to train the subspace method
# * Must output the subspace method's output in the ``forward`` method as the second list element e.g. [prediction, reconstruction]
#
# .. warning::
#
#    Using custom loss is currently only implemented if the second modality is the reconstructed modality, e.g. the image in tabular-image fusion, or the second tabular modality in tabular-tabular fusion.
#
#    The reconstruction shape must be the same as the input shape.
#
# Here's an example:

import numpy as np
import torch
import torch.nn as nn
from fusilli.fusionmodels.base_model import ParentFusionModel


class TemplateSubspaceMethod(ParentFusionModel):
    """
    Template for a subspace-based fusion model that has the subspace method trained before the main prediction model.
    """

    # str: Name of the method.
    method_name = "Template Subspace Method"
    # str: Type of modality.
    modality_type = "tab_img"
    # str: Type of fusion.
    fusion_type = "subspace"

    def __init__(self, pred_type, data_dims, params):
        """
        Parameters
        ----------
        pred_type : str
            Type of prediction to be performed.
        data_dims : list
            List of dimensions of the data.
        params : dict
            Dictionary of parameters.
        """
        super().__init__(pred_type, data_dims, params)

        # nn.Module: Subspace method.
        self.subspace_method_downsample = nn.Sequential(
            nn.Linear(750, 480),
            nn.ReLU(),
            nn.Linear(480, 220),
            nn.ReLU(),
            nn.Linear(220, 88),
        )
        self.subspace_method_upsample = nn.Sequential(
            nn.Linear(88, 220),
            nn.ReLU(),
            nn.Linear(220, 480),
            nn.ReLU(),
            nn.Linear(480, 750),
        )

        # nn.Module: Prediction layers.
        # Concatenating the subspace method's output with the tabular data
        self.pred_model = nn.Sequential(
            nn.Linear(88 + data_dims[0], 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 5),
        )

        self.set_final_pred_layers(input_dim=5)

        # nn.Module: Custom loss function for the reconstruction
        self.custom_loss = nn.MSELoss()

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : list
            List of modalities.

        Returns
        -------
        list
            List of outputs from the model.
        """
        tabular_1 = x[0]
        tabular_2 = x[1]

        # get the subspace method's output
        subspace_output = self.subspace_method_downsample(tabular_2)
        subspace_reconstruction = self.subspace_method_upsample(subspace_output)

        # get the prediction model's output
        out_fused = self.pred_model(torch.cat([x[0], subspace_output]))

        prediction = self.final_prediction(out_fused)

        return [prediction, subspace_reconstruction]

# %%
# Adding model checks and things to make sure this is modifiable.
# Look at Step 3 in :ref:`how_to_contribute_a_template_other_fusion` for more details on this.

# adding calc_fused_layers to make it modifiabel

# %%
# Option 2: how to create a simultaneously-trained subspace-based fusion model
# -----------------------------------------------------------------------------
#
# Have to have a subspace method which is a pytorch lighting module, a class which has train methods, and the fusion model class.
#


# how to create it
