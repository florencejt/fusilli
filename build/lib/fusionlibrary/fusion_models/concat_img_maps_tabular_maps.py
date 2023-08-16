import torch.nn as nn
from fusionlibrary.fusion_models.base_pl_model import ParentFusionModel
import torch


class ConcatImageMapsTabularMaps(ParentFusionModel, nn.Module):
    """
    Concatenating the feature maps of the first tabular modalities and the feature maps of the
    image modality.

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
    img_layers : dict
        Dictionary containing the layers of the image data.
    fused_layers : nn.Sequential
        Sequential layer containing the fused layers.
    final_prediction : nn.Sequential
        Sequential layer containing the final prediction layers. The final prediction layers
          take in the number of
        features of the fused layers as input.

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
        """
        ParentFusionModel.__init__(self, pred_type, data_dims, params)
        self.method_name = "Concatenating clinical feature maps and MRI feature maps"
        self.modality_type = "tab_img"
        self.fusion_type = "operation"
        self.pred_type = pred_type

        self.set_img_layers()
        self.set_mod1_layers()

        self.fused_dim = (
            list(self.mod1_layers.values())[-1][0].out_features
            + list(self.img_layers.values())[-1][0].out_channels
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
        x_tab1 = x[0].squeeze(dim=1)
        x_img = x[1].unsqueeze(dim=1)

        for layer in self.mod1_layers.values():
            x_tab1 = layer(x_tab1)

        for layer in self.img_layers.values():
            x_img = layer(x_img)

        x_img = x_img.view(x_img.size(0), -1)

        out_fuse = torch.cat((x_tab1, x_img), dim=-1)

        out_fuse = self.fused_layers(out_fuse)

        out = self.final_prediction(out_fuse)

        return [
            out,
        ]
