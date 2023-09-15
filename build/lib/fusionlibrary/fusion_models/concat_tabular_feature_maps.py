"""
Concatenating the feature maps of the two tabular modalities.
"""

import torch.nn as nn
from fusionlibrary.fusion_models.base_model import ParentFusionModel
import torch
from torch.autograd import Variable

from fusionlibrary.utils import check_model_validity


class ConcatTabularFeatureMaps(ParentFusionModel, nn.Module):
    """
    Concatenating the feature maps of the two tabular modalities.

    Attributes
    ----------
    pred_type : str
        Type of prediction to be performed.
    mod1_layers : dict
        Dictionary containing the layers of the first modality. Calculated in the
        :meth:`~ParentFusionModel.set_mod1_layers` method.
    mod2_layers : dict
        Dictionary containing the layers of the second modality. Calculated in the
        :meth:`~ParentFusionModel.set_mod2_layers` method.
    fused_dim : int
        Number of features of the fused layers. In this method, it's the size of the tabular 1
        layers output plus the size of the tabular 2 layers output.
    fused_layers : nn.Sequential
        Sequential layer containing the fused layers. Calculated in the
        :meth:`~ConcatTabularFeatureMaps.calc_fused_layers` method.
    final_prediction : nn.Sequential
        Sequential layer containing the final prediction layers. The final prediction layers
        take in the number of features of the fused layers as input. Calculated in the
        :meth:`~ConcatTabularFeatureMaps.calc_fused_layers` method.
    """

    # str: Name of the method.
    method_name = "Concatenating tabular feature maps"
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

        self.fused_dim = (
            list(self.mod1_layers.values())[-1][0].out_features
            + list(self.mod2_layers.values())[-1][0].out_features
        )

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


"""

"""
