"""
Concatenating the input data of the first tabular modality and the feature maps of the
image modality.
"""

import torch.nn as nn
from fusilli.fusionmodels.base_model import ParentFusionModel
import torch
from torch.autograd import Variable

from fusilli.utils import check_model_validity


class ConcatImageMapsTabularData(ParentFusionModel, nn.Module):
    """
    Concatenating the input data of the first tabular modalities and the feature maps of the
    image modality.

    Attributes
    ----------
    prediction_task : str
        Type of prediction to be performed.
    img_layers : nn.ModuleDict
        Dictionary containing the layers of the image data.
    fused_layers : nn.Sequential
        Sequential layer containing the fused layers. Calculated in the
        :meth:`~ConcatImageMapsTabularData.calc_fused_layers` method.
    final_prediction : nn.Sequential
        Sequential layer containing the final prediction layers. The final prediction layers
        take in the number of features of the fused layers as input.
        Calculated in the :meth:`~ConcatImageMapsTabularData.calc_fused_layers` method.
    fused_dim : int
        Number of features of the fused layers. Calculated in the
        :meth:`~ConcatImageMapsTabularData.calc_fused_layers` method.
    """

    #: str: Name of the method.
    method_name = "Concatenating tabular data with image feature maps"
    #: str: Type of modality.
    modality_type = "tabular_image"
    #: str: Type of fusion.
    fusion_type = "operation"

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

        self.set_img_layers()

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
        try:
            dummy_conv_output = Variable(
                torch.rand((1,) + tuple(self.data_dims["img_dim"]))
            )
            for layer in self.img_layers.values():
                dummy_conv_output = layer(dummy_conv_output)
        except:
            pass

        flattened_img_size = dummy_conv_output.data.view(1, -1).size(1)

        self.fused_dim = self.data_dims["mod1_dim"] + flattened_img_size

    def calc_fused_layers(self):
        """
        Calculate the fused layers.

        Returns
        -------
        None
        """

        # ~~ Checks ~~
        check_model_validity.check_dtype(self.img_layers, nn.ModuleDict, "img_layers")

        check_model_validity.check_img_dim(
            self.img_layers, self.data_dims["img_dim"], "img_layers"
        )

        # check fused layers
        self.get_fused_dim()
        self.fused_layers, out_dim = check_model_validity.check_fused_layers(
            self.fused_layers, self.fused_dim
        )

        self.set_final_pred_layers(out_dim)

    def forward(self, x1, x2):
        """
        Forward pass of the model.

        Parameters
        ----------
        x1 : torch.Tensor
            Input tensor for the first tabular modality.
        x2 : torch.Tensor
            Input tensor for the image modality.

        Returns
        -------
        out_pred : torch.Tensor
            Tensor containing the predicted values.
        """

        # ~~ Checks ~~
        check_model_validity.check_model_input(x1)
        check_model_validity.check_model_input(x2)

        x_tab1 = x1.squeeze(dim=1)
        x_img = x2

        for layer in self.img_layers.values():
            x_img = layer(x_img)

        x_img = x_img.view(x_img.size(0), -1)

        out_fuse = torch.cat((x_tab1, x_img), dim=-1)

        out_fuse = self.fused_layers(out_fuse)

        out = self.final_prediction(out_fuse)

        return out
