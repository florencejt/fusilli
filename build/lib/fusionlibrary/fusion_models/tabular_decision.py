"""
Decision fusion of two types of tabular data.
"""

import torch.nn as nn
from fusionlibrary.fusion_models.base_pl_model import ParentFusionModel
import torch


class TabularDecision(ParentFusionModel, nn.Module):
    """
    This class implements a model that fuses the two types of tabular data using a decision fusion
        approach.

    Attributes
    ----------
    mod1_layers : dict
        Dictionary containing the layers of the 1st type of tabular data.
    mod2_layers : dict
        Dictionary containing the layers of the 2nd type of tabular data.
    fused_layers : nn.Sequential
        Sequential layer containing the fused layers.
    final_prediction_tab1 : nn.Sequential
        Sequential layer containing the final prediction layers for the first tabular data.
    final_prediction_tab2 : nn.Sequential
        Sequential layer containing the final prediction layers for the second tabular data.

    """

    # str: Name of the method.
    method_name = "Tabular decision"
    # str: Type of modality.
    modality_type = "both_tab"
    # str: Type of fusion.
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

        self.set_mod1_layers()
        self.set_mod2_layers()
        self.calc_fused_layers()

    def calc_fused_layers(self):
        """
        Calculates the fusion layers.

        Returns
        -------
        None
        """
        tab1_fused_dim = list(self.mod1_layers.values())[-1][0].out_features
        self.set_final_pred_layers(tab1_fused_dim)
        self.final_prediction_tab1 = self.final_prediction

        tab2_fused_dim = list(self.mod2_layers.values())[-1][0].out_features
        self.set_final_pred_layers(tab2_fused_dim)
        self.final_prediction_tab2 = self.final_prediction

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : list
            List containing the two types of tabular data.

        Returns
        -------
        list
            List containing the fused prediction."""
        x_tab1 = x[0]
        x_tab2 = x[1]

        for i, (k, layer) in enumerate(self.mod1_layers.items()):
            x_tab1 = layer(x_tab1)

        for i, (k, layer) in enumerate(self.mod2_layers.items()):
            x_tab2 = layer(x_tab2)

        # predictions for each method
        pred_tab1 = self.final_prediction_tab1(x_tab1)
        pred_tab2 = self.final_prediction_tab2(x_tab2)

        # Combine predictions by averaging them together
        out_fuse = torch.mean(torch.stack([pred_tab1, pred_tab2]), dim=0)

        return [
            out_fuse,
        ]
