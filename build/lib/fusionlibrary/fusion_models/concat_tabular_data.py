"""
Concatenating the two tabular modalities at the data-level (early fusion)
"""

import torch.nn as nn
from fusionlibrary.fusion_models.base_pl_model import ParentFusionModel
import torch


class ConcatTabularData(ParentFusionModel, nn.Module):
    """
    Concatenating the two tabular modalities at the data-level (early fusion)

    Attributes
    ----------
    method_name : str
        Name of the method. (Concatenating tabular data)
    modality_type : str
        Type of modality. (both_tab)
    fusion_type : str
        Type of fusion. (operation)
    pred_type : str
        Type of prediction to be performed.
    mod1_layers : dict
        Dictionary containing the layers of the first modality. Calculated in the
        :meth:`~ParentFusionModel.set_mod1_layers` method.
    mod2_layers : dict
        Dictionary containing the layers of the second modality. Calculated in the
        :meth:`~ParentFusionModel.set_mod2_layers` method.
    fused_layers : nn.Sequential
        Sequential layer containing the fused layers. Calculated in the
        :meth:`~ParentFusionModel.set_fused_layers` method.
    final_prediction : nn.Sequential
        Sequential layer containing the final prediction layers. The final prediction layers
        take in the number of features of the fused layers as input.
        Calculated in the :meth:`~ParentFusionModel.set_final_pred_layers` method.

    Methods
    -------
    forward(x)
        Forward pass of the model.
    """

    method_name = "Concatenating tabular data"
    modality_type = "both_tab"
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
        # super(ParentFusionModel, self).__init__(pred_type, data_dims, params)
        ParentFusionModel.__init__(self, pred_type, data_dims, params)

        self.pred_type = pred_type

        self.set_mod1_layers()
        self.set_mod2_layers()

        self.fused_dim = self.mod1_dim + self.mod2_dim
        self.set_fused_layers(self.fused_dim)
        self.set_final_pred_layers()

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : list
            List containing the data of the two modalities.

        Returns
        -------
        list
            List containing the output of the model.
        """
        x_fuse = torch.cat(x, -1)

        out_fuse = self.fused_layers(x_fuse)

        out_pred = self.final_prediction(out_fuse)

        return [
            out_pred,
        ]
