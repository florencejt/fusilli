"""
Model that fuses the first tabular data and the image data using a decision
fusion approach.
"""

import torch.nn as nn
from fusionlibrary.fusion_models.base_pl_model import ParentFusionModel
import torch
from torch.autograd import Variable


class ImageDecision(ParentFusionModel, nn.Module):
    """
    This class implements a model that fuses the first tabular data and the image data using a decision fusion
        approach.

    Attributes
    ----------
    fusion_type : str
        Type of fusion to be performed.
    modality_type : str
        Type of modalities used.
    method_name : str
        Name of the method.
    mod1_layers : dict
        Dictionary containing the layers of the 1st type of tabular data.
    img_layers : dict
        Dictionary containing the layers of the image data.
    fused_layers : nn.Sequential
        Sequential layer containing the fused layers.
    final_prediction : nn.Sequential
        Sequential layer containing the final prediction layers.

    Methods
    -------
    forward(x)
        Forward pass of the model.
    """

    fusion_type = "decision"
    modality_type = "tab_img"
    method_name = "Image decision fusion"

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

        self.set_img_layers()
        self.set_mod1_layers()
        self.calc_fused_layers()

    def calc_fused_layers(self):
        # ~~ Tabular data ~~

        self.tab_fused_dim = list(self.mod1_layers.values())[-1][0].out_features
        self.set_final_pred_layers(self.tab_fused_dim)
        self.final_prediction_tab1 = self.final_prediction

        # ~~ Image data ~~

        dummy_conv_output = Variable(torch.rand((1,) + tuple(self.data_dims[-1])))
        for layer in self.img_layers.values():
            dummy_conv_output = layer(dummy_conv_output)
        img_fusion_size = dummy_conv_output.data.view(1, -1).size(1)

        self.img_fused_dim = img_fusion_size
        self.set_final_pred_layers(self.img_fused_dim)
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
        x_img = x[1].unsqueeze(dim=1)

        for i, (k, layer) in enumerate(self.mod1_layers.items()):
            x_tab1 = layer(x_tab1)
            x_img = self.img_layers[k](x_img)

        x_img = x_img.view(x_img.size(0), -1)

        # predictions for each method
        pred_tab1 = self.final_prediction_tab1(x_tab1)
        pred_img = self.final_prediction_img(x_img)

        # Combine predictions by averaging them together
        out_fuse = torch.mean(torch.stack([pred_tab1, pred_img]), dim=0)

        return [
            out_fuse,
        ]
