"""
Concatenating the feature maps of the two tabular modalities.
"""

# TODO make 3-tabular data work


import torch.nn as nn
from fusilli.fusionmodels.base_model import ParentFusionModel
import torch

from fusilli.utils import check_model_validity


class ConcatTabularFeatureMaps(ParentFusionModel, nn.Module):
    """
    Concatenating the feature maps of the two tabular modalities.

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
    mod3_layers : nn.ModuleDict
        Dictionary containing the layers of the third modality. Calculated in the
        :meth:`~ParentFusionModel.set_mod3_layers` method. None if three modalities are not provided.
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

    #: str: Name of the method.
    method_name = "Concatenating tabular feature maps"
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

        self.set_mod1_layers()
        self.set_mod2_layers()
        if self.data_dims["mod3_dim"] is not None:
            self.set_mod3_layers()

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
        if self.data_dims["mod3_dim"] is not None:
            self.fused_dim += list(self.mod3_layers.values())[-1][0].out_features

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
        if self.data_dims["mod3_dim"] is not None:
            check_model_validity.check_dtype(
                self.mod3_layers, nn.ModuleDict, "mod3_layers"
            )

        self.get_fused_dim()
        self.fused_layers, out_dim = check_model_validity.check_fused_layers(
            self.fused_layers, self.fused_dim
        )

        # setting final prediction layers with final out features of fused layers
        self.set_final_pred_layers(out_dim)

    def forward(self, x1, x2, x3=None):
        """
        Forward pass of the model.

        Parameters
        ----------
        x1 : torch.Tensor
            Input tensor for the first modality.
        x2 : torch.Tensor
            Input tensor for the second modality.
        x3 : torch.Tensor or None
            Input tensor for the third modality. Default is None.

        Returns
        -------
        torch.Tensor
            Fused prediction.
        """

        # ~~ Checks ~~
        check_model_validity.check_model_input(x1)
        check_model_validity.check_model_input(x2)
        if x3 is not None:
            check_model_validity.check_model_input(x3)

        for layer in self.mod1_layers.values():
            x1 = layer(x1)

        for layer in self.mod2_layers.values():
            x2 = layer(x2)

        # x1 and x2 to a tuple
        x_tuple = (x1, x2)

        if x3 is not None:
            for layer in self.mod3_layers.values():
                x3 = layer(x3)
            x_tuple = (x1, x2, x3)

        out_fuse = torch.cat(x_tuple, dim=-1)

        out_fuse = self.fused_layers(out_fuse)

        out = self.final_prediction(out_fuse)

        return out


"""

"""
