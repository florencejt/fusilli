"""
Concatenating the input data of the first tabular modality and the feature maps of the
image modality.
"""

import torch.nn as nn
from fusilli.fusionmodels.base_model import ParentFusionModel
import torch
from torch.autograd import Variable

from fusilli.utils import check_model_validity


class ConcatImageMapsTabularData(ParentFusionModel, nn.Module):
    """
    Concatenating the input data of the first tabular modalities and the feature maps of the
    image modality.

    Attributes
    ----------
    pred_type : str
        Type of prediction to be performed.
    img_layers : dict
        Dictionary containing the layers of the image data.
    fused_layers : nn.Sequential
        Sequential layer containing the fused layers. Calculated in the
        :meth:`~ConcatImageMapsTabularData.calc_fused_layers` method.
    final_prediction : nn.Sequential
        Sequential layer containing the final prediction layers. The final prediction layers
        take in the number of features of the fused layers as input.
        Calculated in the :meth:`~ConcatImageMapsTabularData.calc_fused_layers` method.
    fused_dim : int
        Number of features of the fused layers. Calculated in the
        :meth:`~ConcatImageMapsTabularData.calc_fused_layers` method.
    """

    #: str: Name of the method.
    method_name = "Concatenating tabular data with image feature maps"
    #: str: Type of modality.
    modality_type = "tab_img"
    #: str: Type of fusion.
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

        self.get_fused_dim()

        self.set_fused_layers(self.fused_dim)

        self.calc_fused_layers()

    def get_fused_dim(self):
        """
        Get the number of features of the fused layers.

        Returns
        -------
        None
        """

        dummy_conv_output = Variable(torch.rand((1,) + tuple(self.img_dim)))
        for layer in self.img_layers.values():
            dummy_conv_output = layer(dummy_conv_output)
        flattened_img_size = dummy_conv_output.data.view(1, -1).size(1)

        self.fused_dim = self.mod1_dim + flattened_img_size

    def calc_fused_layers(self):
        """
        Calculate the fused layers.

        Returns
        -------
        None
        """

        # ~~ Checks ~~
        check_model_validity.check_dtype(self.img_layers, nn.ModuleDict, "img_layers")

        check_model_validity.check_img_dim(self.img_layers, self.img_dim, "img_layers")

        # check fused layers
        self.get_fused_dim()
        self.fused_layers, out_dim = check_model_validity.check_fused_layers(
            self.fused_layers, self.fused_dim
        )

        self.set_final_pred_layers(out_dim)

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

        for layer in self.img_layers.values():
            x_img = layer(x_img)

        x_img = x_img.view(x_img.size(0), -1)

        out_fuse = torch.cat((x_tab1, x_img), dim=-1)

        out_fuse = self.fused_layers(out_fuse)

        out = self.final_prediction(out_fuse)

        return [
            out,
        ]
