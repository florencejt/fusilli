"""
Tabular1 uni-modal model.
"""

import torch.nn as nn
from fusionlibrary.fusion_models.base_pl_model import ParentFusionModel


class Tabular1Unimodal(ParentFusionModel, nn.Module):
    """Uni-modal model for tabular data.

    This class implements a uni-modal model using only the 1st type of tabular data.

    Attributes
    ----------
    mod1_layers : dict
        Dictionary containing the layers of the 1st type of tabular data.
    fused_layers : nn.Sequential
        Sequential layer containing the fused layers.
    final_prediction : nn.Sequential
        Sequential layer containing the final prediction layers.
    fused_dim : int
        Dimension of the fused layer.
    """

    #: str: Name of the method.
    method_name = "Tabular1 uni-modal"

    #: str: Modality type.
    modality_type = "tabular1"

    #: str: Fusion type.
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

        Raises
        ------
        ValueError
            If the prediction type is not valid.
        """

        ParentFusionModel.__init__(self, pred_type, data_dims, params)

        self.pred_type = pred_type

        self.set_mod1_layers()
        self.calc_fused_layers()

    def calc_fused_layers(self):
        """Calculate the fused layers.

        If the mod1_layers are modified, this function will recalculate the fused layers.
        """
        self.fused_dim = list(self.mod1_layers.values())[-1][0].out_features
        self.set_fused_layers(self.fused_dim)
        self.set_final_pred_layers()

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        list
            List containing the output of the model.
        """
        x_tab1 = x

        for layer in self.mod1_layers.values():
            x_tab1 = layer(x_tab1)

        out_fuse = self.fused_layers(x_tab1)

        out_pred = self.final_prediction(out_fuse)

        return [
            out_pred,
        ]
