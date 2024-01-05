"""
Activation-function fusion model for tabular data.
"""

import torch.nn as nn
from fusilli.fusionmodels.base_model import ParentFusionModel
import torch

from fusilli.utils import check_model_validity


class ActivationFusion(ParentFusionModel, nn.Module):
    """
    Performs an element wise product of the feature maps of the two tabular modalities, 
    tanh activation function and sigmoid activation function. Afterwards the the first tabular modality feature
    map is concatenated with the fused feature map. 

    Attributes
    ----------
    prediction_task : str
        Type of prediction to be performed.
    mod1_layers : nn.ModuleDict
        Dictionary containing the layers of the first modality. Calculated in the
        :meth:`~ParentFusionModel.set_mod1_layers` method.
    mod2_layers : nn.ModuleDict
        Dictionary containing the layers of the second modality. Calculated in the
        :meth:`~ParentFusionModel.set_mod2_layers` method.
    fused_dim : int
        Number of features of the fused layers. In this method, it's the size of the tabular 1
        layers output plus the size of the tabular 2 layers output.
    fused_layers : nn.Sequential
        Sequential layer containing the fused layers. Calculated in the
        :meth:`~ActivationFusion.calc_fused_layers` method.
    final_prediction : nn.Sequential
        Sequential layer containing the final prediction layers. The final prediction layers
        take in the number of features of the fused layers as input. Calculated in the
        :meth:`~ActivationFusion.calc_fused_layers` method.
    """

    #: str: Name of the method.
    method_name = "Activation function map fusion"
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
            Number of classes in the multiclass classification problem.
        """
        ParentFusionModel.__init__(self, prediction_task, data_dims, multiclass_dimensions)

        self.prediction_task = prediction_task

        self.set_mod1_layers()
        self.set_mod2_layers()

        self.get_fused_dim()
        self.set_fused_layers(self.fused_dim)

        self.calc_fused_layers()

    def get_fused_dim(self):
        """
        Get the number of features of the fused layers.
        Assuming mod1_layers and mod2_layers output the same dimension.
        """
        mod1_output_dim = list(self.mod1_layers.values())[-1][0].out_features
        mod2_output_dim = list(self.mod2_layers.values())[-1][0].out_features
        # New fused dimension is the sum of mod1 and mod2 output dimensions
        self.fused_dim = mod1_output_dim + mod2_output_dim

    def calc_fused_layers(self):
        """
        Calculate the fused layers.

        Returns
        -------
        None
        """

        # ~~ Checks ~~
        check_model_validity.check_dtype(self.mod1_layers, nn.ModuleDict, "mod1_layers")
        check_model_validity.check_dtype(self.mod2_layers, nn.ModuleDict, "mod2_layers")

        mod1_output_dim = list(self.mod1_layers.values())[-1][0].out_features
        mod2_output_dim = list(self.mod2_layers.values())[-1][0].out_features
        if mod1_output_dim != mod2_output_dim:
            raise UserWarning(
                "The number of output features of mod1_layers and mod2_layers must be the same for Activation fusion. Please change the final layers in the modality layers to have the same number of output features as each other.")

        self.get_fused_dim()
        self.fused_layers, out_dim = check_model_validity.check_fused_layers(
            self.fused_layers, self.fused_dim
        )

        # setting final prediction layers with final out features of fused layers
        self.set_final_pred_layers(out_dim)

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : tuple
            Tuple containing the input data.

        Returns
        -------
        list
            List containing the output of the model.
        """

        # ~~ Checks ~~
        check_model_validity.check_model_input(x)

        x_tab1 = x[0]
        x_tab2 = x[1]

        for layer in self.mod1_layers.values():
            x_tab1 = layer(x_tab1)

        for layer in self.mod2_layers.values():
            x_tab2 = layer(x_tab2)

        x_tab1 = torch.squeeze(x_tab1, 1)
        x_tab2 = torch.squeeze(x_tab2, 1)

        out_fuse = torch.mul(x_tab1, x_tab2)

        out_fuse = torch.tanh(out_fuse)
        out_fuse = torch.sigmoid(out_fuse)

        out_fuse = torch.cat((out_fuse, x_tab1), dim=1)

        out_fuse = self.fused_layers(out_fuse)

        out = self.final_prediction(out_fuse)

        return [
            out,
        ]


"""

"""
