import torch.nn as nn
from fusionlibrary.fusion_models.base_pl_model import ParentFusionModel
import torch


class ConcatTabularFeatureMaps(ParentFusionModel, nn.Module):
    """
    Concatenating the feature maps of the two tabular modalities.

    Attributes
    ----------
    method_name : str
        Name of the method.
    modality_type : str
        Type of modality.
    fusion_type : str
        Type of fusion.
    pred_type : str
        Type of prediction to be performed.
    mod1_layers : dict
        Dictionary containing the layers of the first modality.
    mod2_layers : dict
        Dictionary containing the layers of the second modality.
    fused_layers : nn.Sequential
        Sequential layer containing the fused layers.
    final_prediction : nn.Sequential
        Sequential layer containing the final prediction layers. The final prediction layers
        take in the number of features of the fused layers as input.

    Methods
    -------
    forward(x)
        Forward pass of the model.
    """

    method_name = "Concatenating tabular feature maps"
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
        ParentFusionModel.__init__(self, pred_type, data_dims, params)

        self.pred_type = pred_type

        self.set_mod1_layers()
        self.set_mod2_layers()

        self.fused_dim = (
            list(self.mod1_layers.values())[-1][0].out_features
            + list(self.mod2_layers.values())[-1][0].out_features
        )
        self.set_fused_layers(self.fused_dim)
        self.set_final_pred_layers()

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

        x_tab1 = x[0]
        x_tab2 = x[1]

        for layer in self.mod1_layers.values():
            x_tab1 = layer(x_tab1)

        for layer in self.mod2_layers.values():
            x_tab2 = layer(x_tab2)

        out_fuse = torch.cat((x_tab1, x_tab2), dim=-1)

        out_fuse = self.fused_layers(out_fuse)

        out = self.final_prediction(out_fuse)

        return [
            out,
        ]
