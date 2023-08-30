"""
Concatenating the input data of the first tabular modality and the feature maps of the
image modality.
"""

import torch.nn as nn
from fusionlibrary.fusion_models.base_pl_model import ParentFusionModel
import torch
from torch.autograd import Variable


class ConcatImageMapsTabularData(ParentFusionModel, nn.Module):
    """
    Concatenating the input data of the first tabular modalities and the feature maps of the
    image modality.

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
    img_layers : dict
        Dictionary containing the layers of the image data.
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

    method_name = "Concatenating tabular data with image feature maps"
    modality_type = "tab_img"
    fusion_type = "operation"

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
        self.calc_fused_layers()

        # self.fused_dim = (
        #     self.mod1_dim + list(self.img_layers.values())[-1][0].out_channels
        # )
        # self.set_fused_layers(self.fused_dim)
        # self.set_final_pred_layers()

    def calc_fused_layers(self):
        # get flattened image output dim
        dummy_conv_output = Variable(torch.rand((1,) + tuple(self.data_dims[-1])))
        for layer in self.img_layers.values():
            dummy_conv_output = layer(dummy_conv_output)
        flattened_img_size = dummy_conv_output.data.view(1, -1).size(1)

        self.fused_dim = self.mod1_dim + flattened_img_size
        self.set_fused_layers(self.fused_dim)
        self.set_final_pred_layers()

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

        for layer in self.img_layers.values():
            x_img = layer(x_img)

        x_img = x_img.view(x_img.size(0), -1)

        out_fuse = torch.cat((x_tab1, x_img), dim=-1)

        out_fuse = self.fused_layers(out_fuse)

        out = self.final_prediction(out_fuse)

        return [
            out,
        ]
