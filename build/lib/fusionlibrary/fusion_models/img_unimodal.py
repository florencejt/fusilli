"""
Uni-modal model using only the image data.
"""

import torch.nn as nn
from fusionlibrary.fusion_models.base_model import ParentFusionModel
from torch.autograd import Variable
import torch
from fusionlibrary.utils import check_model_validity


class ImgUnimodal(ParentFusionModel, nn.Module):
    """
    This class implements a uni-modal model using only the image data.

    Attributes
    ----------
    img_layers : dict
        Dictionary containing the layers of the image data.
    fused_dim : int
        Number of features of the fused layers. This is the flattened output size of the
        image layers.
    fused_layers : nn.Sequential
        Sequential layer containing the fused layers.
    final_prediction : nn.Sequential
        Sequential layer containing the final prediction layers.
    """

    # str: Name of the method.
    method_name = "Image unimodal"
    # str: Type of modality.
    modality_type = "img"
    # str: Type of fusion.
    fusion_type = "Uni-modal"

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
        flattened_img_output_size = dummy_conv_output.data.view(1, -1).size(1)

        self.fused_dim = flattened_img_output_size

    def calc_fused_layers(self):
        """
        Calculates the fused layers.

        Returns
        -------
        None
        """
        # check fused layers
        self.fused_layers, out_dim = check_model_validity.check_fused_layers(
            self.fused_layers, self.fused_dim
        )

        # check img layers
        check_model_validity.check_dtype(self.img_layers, nn.ModuleDict, "img_layers")
        check_model_validity.check_img_dim(self.img_layers, self.img_dim, "img_layers")

        self.set_final_pred_layers(out_dim)

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Tensor containing the image data.

        Returns
        -------
        out_pred : list
            List containing the predictions.
        """
        x_img = x

        for layer in self.img_layers.values():
            x_img = layer(x_img)

        x_img = x_img.view(x_img.size(0), -1)

        out_fuse = self.fused_layers(x_img)

        out_pred = self.final_prediction(out_fuse)

        return [
            out_pred,
        ]
