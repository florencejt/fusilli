"""
Decision fusion of two types of tabular data.
"""

# TODO make 3-tabular data work


import torch.nn as nn
from fusilli.fusionmodels.base_model import ParentFusionModel
import torch

from fusilli.utils import check_model_validity


class TabularDecision(ParentFusionModel, nn.Module):
    """
    This class implements a model that fuses the two types of tabular data using a decision fusion
        approach.

    Attributes
    ----------
    mod1_layers : nn.ModuleDict
        Dictionary containing the layers of the 1st type of tabular data.
    mod2_layers : nn.ModuleDict
        Dictionary containing the layers of the 2nd type of tabular data.
    fused_layers : nn.Sequential
        Sequential layer containing the fused layers.
    final_prediction_tab1 : nn.Sequential
        Sequential layer containing the final prediction layers for the first tabular data.
    final_prediction_tab2 : nn.Sequential
        Sequential layer containing the final prediction layers for the second tabular data.
    fusion_operation : function
        Function that performs the fusion operation. Default is torch.mean(torch.stack([x, y]), dim=0).

    """

    #: str: Name of the method.
    method_name = "Tabular decision"
    #: str: Type of modality.
    modality_type = "tabular_tabular"
    #: str: Type of fusion.
    fusion_type = "operation"
    #: str: Available for three tabular modalities.
    three_modalities = True

    def __init__(self, prediction_task, data_dims, multiclass_dimensions):
        """
        Parameters
        ----------
        prediction_task : str
            Type of prediction to be performed.
        data_dims : dict
            Dictionary of data dimensions with keys "mod1_dim", "mod2_dim", "mod3_dim", and "img_dim".
        multiclass_dimensions : int
            Number of classes in the multiclass classification task.
        """
        ParentFusionModel.__init__(
            self, prediction_task, data_dims, multiclass_dimensions
        )

        self.prediction_task = prediction_task

        self.fusion_operation = lambda x: torch.mean(torch.stack(x), dim=0)

        self.set_mod1_layers()
        self.set_mod2_layers()

        if self.data_dims["mod3_dim"] is not None:
            self.set_mod3_layers()

        self.calc_fused_layers()

    def calc_fused_layers(self):
        """
        Calculates the fusion layers.

        Returns
        -------
        None
        """

        check_model_validity.check_var_is_function(
            self.fusion_operation, "fusion_operation"
        )
        check_model_validity.check_dtype(self.mod1_layers, nn.ModuleDict, "mod1_layers")
        check_model_validity.check_dtype(self.mod2_layers, nn.ModuleDict, "mod2_layers")

        # Set the final prediction layers for each modality: tab1
        tab1_fused_dim = list(self.mod1_layers.values())[-1][0].out_features
        self.set_final_pred_layers(tab1_fused_dim)
        self.final_prediction_tab1 = self.final_prediction

        # Set the final prediction layers for each modality: tab2
        tab2_fused_dim = list(self.mod2_layers.values())[-1][0].out_features
        self.set_final_pred_layers(tab2_fused_dim)
        self.final_prediction_tab2 = self.final_prediction

        if self.data_dims["mod3_dim"] is not None:
            # Check if the mod3_layers attribute is a ModuleDict
            check_model_validity.check_dtype(
                self.mod3_layers, nn.ModuleDict, "mod3_layers"
            )
            # Set the final prediction layers for each modality: tab3
            tab3_fused_dim = list(self.mod3_layers.values())[-1][0].out_features
            self.set_final_pred_layers(tab3_fused_dim)
            self.final_prediction_tab3 = self.final_prediction

    def forward(self, x1, x2, x3=None):
        """
        Forward pass of the model.

        Parameters
        ----------
        x1 : torch.Tensor
            Input tensor for the first modality.
        x2 : torch.Tensor
            Input tensor for the second modality.
        x3 : torch.Tensor
            Input tensor for the third modality. Default is None.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """

        # ~~ Checks ~~
        check_model_validity.check_model_input(x1)
        check_model_validity.check_model_input(x2)
        if x3 is not None:
            check_model_validity.check_model_input(x3)

        for i, (k, layer) in enumerate(self.mod1_layers.items()):
            x1 = layer(x1)

        for i, (k, layer) in enumerate(self.mod2_layers.items()):
            x2 = layer(x2)

        if x3 is not None:
            for i, (k, layer) in enumerate(self.mod3_layers.items()):
                x3 = layer(x3)

        # predictions for each method
        pred_tab1 = self.final_prediction_tab1(x1)
        pred_tab2 = self.final_prediction_tab2(x2)
        preds = [pred_tab1, pred_tab2]
        if x3 is not None:
            pred_tab3 = self.final_prediction_tab3(x3)
            preds.append(pred_tab3)

        # Combine predictions by averaging them together
        out_fuse = self.fusion_operation(preds)

        return out_fuse
