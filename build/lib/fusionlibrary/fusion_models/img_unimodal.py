"""
Uni-modal model using only the image data.
"""

import torch.nn as nn
from fusionlibrary.fusion_models.base_pl_model import ParentFusionModel


class ImgUnimodal(ParentFusionModel, nn.Module):
    """
    This class implements a uni-modal model using only the image data.

    Attributes
    ----------
    method_name : str
        Name of the method.
    modality_type : str
        Type of modality.
    fusion_type : str
        Type of fusion.
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

    method_name = "Image unimodal"
    modality_type = "img"
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

        self.fused_dim = list(self.img_layers.values())[-1][0].out_channels
        self.set_fused_layers(self.fused_dim)
        self.set_final_pred_layers()

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
        x_img = x.unsqueeze(dim=1)

        for layer in self.img_layers.values():
            x_img = layer(x_img)

        x_img = x_img.view(x_img.size(0), -1)

        out_fuse = self.fused_layers(x_img)

        out_pred = self.final_prediction(out_fuse)

        return [
            out_pred,
        ]
