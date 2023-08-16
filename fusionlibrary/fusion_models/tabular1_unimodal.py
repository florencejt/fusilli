import torch.nn as nn
from fusionlibrary.fusion_models.base_pl_model import ParentFusionModel


class Tabular1Unimodal(ParentFusionModel, nn.Module):
    """
    This class implements a uni-modal model using only the 1st type of tabular data.

    Attributes
    ----------
    method_name : str
        Name of the method.
    modality_type : str
        Type of modality.
    fusion_type : str
        Type of fusion.
    mod1_layers : dict
        Dictionary containing the layers of the 1st type of tabular data.
    fused_layers : nn.Sequential
        Sequential layer containing the fused layers.
    final_prediction : nn.Sequential
        Sequential layer containing the final prediction layers.
    fused_dim : int
        Dimension of the fused layer.

    Methods
    -------
    forward(x)
        Forward pass of the model.
    """

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
        self.method_name = "Clinical only"
        self.modality_type = "tabular1"
        self.fusion_type = "Uni-modal"
        self.pred_type = pred_type

        self.set_mod1_layers()

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
