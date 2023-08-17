import torch.nn as nn
from fusionlibrary.fusion_models.base_pl_model import ParentFusionModel


class Tabular2Unimodal(ParentFusionModel, nn.Module):
    """
    This class implements a uni-modal model using only the 2nd type of tabular data.

    Attributes
    ----------
    method_name : str
        Name of the method.
    modality_type : str
        Type of modality.
    fusion_type : str
        Type of fusion.
    mod2_layers : dict
        Dictionary containing the layers of the 2nd type of tabular data.
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

    method_name = "Tabular2 uni-modal"
    modality_type = "tabular2"
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

        self.set_mod2_layers()

        self.fused_dim = list(self.mod2_layers.values())[-1][0].out_features
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
        x_tab2 = x

        for layer in self.mod2_layers.values():
            x_tab2 = layer(x_tab2)

        out_fuse = self.fused_layers(x_tab2)

        out_pred = self.final_prediction(out_fuse)

        return [
            out_pred,
        ]
