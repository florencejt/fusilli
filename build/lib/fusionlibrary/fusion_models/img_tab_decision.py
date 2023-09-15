"""
Model that fuses the first tabular data and the image data using a decision
fusion approach.
"""

import torch.nn as nn
from fusionlibrary.fusion_models.base_model import ParentFusionModel
import torch
from torch.autograd import Variable

from fusionlibrary.utils import check_model_validity


class ImageDecision(ParentFusionModel, nn.Module):
    """
    This class implements a model that fuses the first tabular data and the image data using a decision fusion
        approach.

    Attributes
    ----------
    mod1_layers : dict
        Dictionary containing the layers of the 1st type of tabular data.
    img_layers : dict
        Dictionary containing the layers of the image data.
    fused_layers : nn.Sequential
        Sequential layer containing the fused layers.
    final_prediction_tab1 : nn.Sequential
        Sequential layer containing the final prediction layers for the first tabular data.
    final_prediction_img : nn.Sequential
        Sequential layer containing the final prediction layers for the image data.
    fusion_operation : function
        Function that performs the fusion operation. Default is torch.mean(torch.stack([x, y]), dim=0).

    .. warning::
        `fusion_operation` should be done on the first dimension, i.e. the batch dimension.
        For example, `lambda x: torch.mean(x, dim=1)`.
        The predictions of the different modalities are stacked on the first dimension before
        `fusion_operation`.


    """

    # str: Name of the method.
    method_name = "Image decision fusion"
    # str: Type of modality.
    modality_type = "tab_img"
    # str: Type of fusion.
    fusion_type = "decision"

    def __init__(self, pred_type, data_dims, params):
        """
        Parameters
        ----------
        pred_type : str
            Type of prediction to be performed.
        data_dims : dict
            Dictionary containing the dimensions of the data.
        params : dict
            Dictionary containing the parameters of the model.
        """
        ParentFusionModel.__init__(self, pred_type, data_dims, params)

        self.pred_type = pred_type

        self.fusion_operation = lambda x: torch.mean(x, dim=1)

        self.set_img_layers()
        self.set_mod1_layers()
        self.calc_fused_layers()

    def calc_fused_layers(self):
        """
        Calculates the fusion layers.

        Returns
        -------
        None
        """

        check_model_validity.check_var_is_function(
            self.fusion_operation, "fusion_operation"
        )
        check_model_validity.check_dtype(self.img_layers, nn.ModuleDict, "img_layers")
        check_model_validity.check_dtype(self.mod1_layers, nn.ModuleDict, "mod1_layers")

        check_model_validity.check_img_dim(self.img_layers, self.img_dim, "img_layers")

        # ~~ Tabular data ~~

        tab_fused_dim = list(self.mod1_layers.values())[-1][0].out_features
        self.set_final_pred_layers(tab_fused_dim)
        self.final_prediction_tab1 = self.final_prediction

        # ~~ Image data ~~

        dummy_conv_output = Variable(torch.rand((1,) + tuple(self.img_dim)))
        for layer in self.img_layers.values():
            dummy_conv_output = layer(dummy_conv_output)
        img_fusion_size = dummy_conv_output.data.view(1, -1).size(1)
        self.set_final_pred_layers(img_fusion_size)
        self.final_prediction_img = self.final_prediction

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : list
            List containing the input data.

        Returns
        -------
        list
            List containing the output of the model.
        """
        x_tab1 = x[0].squeeze(dim=1)
        x_img = x[1]

        for i, (k, layer) in enumerate(self.mod1_layers.items()):
            x_tab1 = layer(x_tab1)

        for i, (k, layer) in enumerate(self.img_layers.items()):
            x_img = layer(x_img)

        x_img = x_img.view(x_img.size(0), -1)

        # predictions for each method
        pred_tab1 = self.final_prediction_tab1(x_tab1)
        pred_img = self.final_prediction_img(x_img)

        # Combine predictions by averaging them together
        print("pred_tab1", pred_tab1)
        print("pred_img", pred_img)
        fusion_input = torch.stack((pred_tab1, pred_img), dim=1)
        print("fusion_input", fusion_input)
        out_fuse = self.fusion_operation(fusion_input)
        print("out_fuse", out_fuse)
        # out_fuse = torch.mean(torch.stack([pred_tab1, pred_img]), dim=0)

        return [
            out_fuse,
        ]
