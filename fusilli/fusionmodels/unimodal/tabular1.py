"""
Tabular1 uni-modal model.
"""

import torch.nn as nn
from fusilli.fusionmodels.base_model import ParentFusionModel
from fusilli.utils import check_model_validity


class Tabular1Unimodal(ParentFusionModel, nn.Module):
    """unimodal model for tabular data.

    This class implements a uni-modal model using only the 1st type of tabular data.

    Attributes
    ----------
    mod1_layers : nn.ModuleDict
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
    fusion_type = "unimodal"

    def __init__(self, prediction_task, data_dims, multiclass_dimensions):
        """
        Parameters
        ----------
        prediction_task : str
            Type of prediction to be performed.
        data_dims : list
            List containing the dimensions of the data.
        multiclass_dimensions : int
            Number of classes in the multiclass classification task.
        """
        ParentFusionModel.__init__(self, prediction_task, data_dims, multiclass_dimensions)

        self.prediction_task = prediction_task

        self.set_mod1_layers()

        self.get_fused_dim()
        self.set_fused_layers(self.fused_dim)

        self.calc_fused_layers()

    def get_fused_dim(self):
        """Get the number of features of the fused layers.

        Returns
        -------
        None
        """

        self.fused_dim = list(self.mod1_layers.values())[-1][0].out_features

    def calc_fused_layers(self):
        """Calculate the fused layers.

        If the mod1_layers are modified, this function will recalculate the fused layers.
        """

        check_model_validity.check_dtype(self.mod1_layers, nn.ModuleDict, "mod1_layers")

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
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        list
            List containing the output of the model.
        """

        check_model_validity.check_model_input(x, uni_modal_flag=True)

        x_tab1 = x
        for layer in self.mod1_layers.values():
            x_tab1 = layer(x_tab1)

        out_fuse = self.fused_layers(x_tab1)

        out_pred = self.final_prediction(out_fuse)

        return [
            out_pred,
        ]
