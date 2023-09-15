"""
Operation / attention / tensor fusion template
===============================================

This is a template for creating your own fusion model: operation-based, attention-based, or tensor-based.
If you want to implement a graph-based or subspace-based fusion model, please refer to the other templates. They require additional functions to be implemented.

There are two ways to create your own fusion model:

1. You can use the preset layers in the :class:`~fusionlibrary.fusion_models.base_model.ParentFusionModel` class. This is the easiest way to create your own fusion model. You can see an example of this in the :class:`~fusionlibrary.fusion_models.concat_tabular_data.ConcatTabularData` class.
2. You can create your own layers. This is the most flexible way to create your own fusion model.

The most important thing to remember is to output the final prediction as a list. This is because in subspace methods, the output is the label output and a reconstruction. This project's architecture is designed to be flexible enough to handle subspace methods too, hence the need for a list.

Have fun creating your own fusion model!
"""

import torch.nn as nn
import torch

# importing the parent fusion model class
from fusionlibrary.fusion_models.base_model import ParentFusionModel


class TemplateFusionModel(ParentFusionModel, nn.Module):
    """
    Do you want to create your own multi-modal fusion model? This is the template for you!

    Attributes
    ----------
    method_name : str
        Name of the method.
    modality_type : str
        Type of modality.
    fusion_type : str
        Type of fusion.
    pred_type : str
        Type of prediction to be performed.
    mod1_layers : dict
        Dictionary containing the layers of the first modality.
    mod2_layers : dict
        Dictionary containing the layers of the second modality.
    fused_layers : nn.Sequential
        Sequential layer containing the fused layers.
    final_prediction : nn.Sequential
        Sequential layer containing the final prediction layers. The final prediction layers
        take in the number of features of the fused layers as input.

    Methods
    -------
    forward(x)
        Forward pass of the model.

    """

    method_name = "Template fusion model"
    modality_type = "both_tab"  # or "tabular1", "tabular2", "both_tab", "tab_img"
    fusion_type = "attention"  # or "operation", "tensor", "graph", "subspace"

    def __init__(self, pred_type, data_dims, params):
        ParentFusionModel.__init__(self, pred_type, data_dims, params)
        self.pred_type = pred_type

        ################################
        # SETTING THE UNI-MODAL LAYERS #
        ################################
        # You can either set the layers to be consistent with the rest of the library
        # or you can set your own layers.

        # USING PARENTFUSIONMODEL PRESET LAYERS
        self.set_mod1_layers()  # set the layers for the first tabular modality
        self.set_mod2_layers()  # set the layers for the second tabular modality
        self.set_img_layers()  # set the layers for the image modality (if using)

        ################################
        #   SETTING THE FUSED LAYERS   #
        ################################

        # Setting a fused dimension: how many features are there after the fusion?
        # e.g. concatenating two tabular modalities would be the sum of the number of features
        self.fused_dim = self.mod1_dim + self.mod2_dim

        # Setting the fused layers: how do you want to fuse the modalities?
        # Again, you can either set the layers to be consistent with the rest of the library
        # or you can set your own layers.

        # USING PARENTFUSIONMODEL PRESET LAYERS
        self.set_fused_layers(self.fused_dim)

        #################################
        # SETTING THE FINAL PRED LAYERS #
        #################################

        # Setting the final prediction layers: how do you want to make the final prediction?
        self.set_final_pred_layers(input_dim=self.fused_dim)
        # Default input dim to final_pred_layers is 64, but you can change this in the function call.

    def forward(self, x):
        """
        Forward pass of the model. This is an example of a concatenation of feature maps!
        Feel free to change this to suit your model - get creative!!

        Parameters
        ----------
        x : list
            List containing the input data.

        Returns
        -------
        list
            List containing the output of the model.
        """

        x_tab1 = x[0]  # tabular1 data
        x_tab2 = x[1]  # tabular2 data

        # pass the data through the modality layers
        for i, (k, layer) in enumerate(self.mod1_layers.items()):
            x_tab1 = layer(x_tab1)
            x_tab2 = self.mod2_layers[k](x_tab2)

        # pass the data through the fused layers
        out_fuse = torch.cat((x_tab1, x_tab2), dim=-1)
        out_fuse = self.fused_layers(out_fuse)

        # pass the data through the final prediction layers
        out = self.final_prediction(out_fuse)

        # You have to return the output of the model as a list.
        # This is because in subspace methods, the output is the label output and a reconstruction.
        # This project's architecture is designed to be flexible enough to handle subspace methods too, hence the list.

        return [
            out,
        ]