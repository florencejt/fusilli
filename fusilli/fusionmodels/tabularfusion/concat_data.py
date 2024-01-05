"""
Concatenating the two tabular modalities at the data-level (early fusion)
"""

from fusilli.fusionmodels.base_model import ParentFusionModel
from fusilli.utils import check_model_validity

import torch
import torch.nn as nn


class ConcatTabularData(ParentFusionModel, nn.Module):
    """
    Concatenating the two tabular modalities at the data-level (early fusion)

    Attributes
    ----------
    prediction_task : str
        Type of prediction to be performed.
    fused_dim : int
        Number of features of the fused layers. In this method, it's the tabular 1 dimension plus
        the tabular 2 dimension.
    fused_layers : nn.Sequential
        Sequential layer containing the fused layers. Calculated in the
        :meth:`~ParentFusionModel.set_fused_layers` method.
    final_prediction : nn.Sequential
        Sequential layer containing the final prediction layers. The final prediction layers
        take in the number of features of the fused layers as input.
        Calculated in the :meth:`~ParentFusionModel.set_final_pred_layers` method.
    """

    #: str: Name of the method.
    method_name = "Concatenating tabular data"
    #: str: Type of modality.
    modality_type = "tabular_tabular"
    #: str: Type of fusion.
    fusion_type = "operation"

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
        self.fused_dim = self.mod1_dim + self.mod2_dim

    def calc_fused_layers(self):
        """
        Calculate the fused layers.

        Returns
        -------
        None
        """

        # check fused layer
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
        x : tuple
            Tuple containing the data of the two modalities.

        Returns
        -------
        list
            List containing the output of the model.
        """

        # ~~ Checks ~~
        check_model_validity.check_model_input(x)

        x_fuse = torch.cat(x, -1)

        out_fuse = self.fused_layers(x_fuse)

        out_pred = self.final_prediction(out_fuse)

        return [
            out_pred,
        ]
