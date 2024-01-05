"""
Using activation functions to fuse tabular data, with self-attention on the second tabular modality.
"""

import torch.nn as nn

from fusilli.fusionmodels.base_model import ParentFusionModel
import torch

from fusilli.utils import check_model_validity


class AttentionAndSelfActivation(ParentFusionModel, nn.Module):
    """
    Applies an attention mechanism on the second tabular modality features and performs an element wise product of
    the feature maps of the two tabular modalities, 
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
        :meth:`~AttentionAndActivation.calc_fused_layers` method.
    final_prediction : nn.Sequential
        Sequential layer containing the final prediction layers. The final prediction layers
        take in the number of features of the fused layers as input. Calculated in the
        :meth:`~AttentionAndActivation.calc_fused_layers` method.
    attention_reduction_ratio : int
        Reduction ratio of the channel attention module.
    """

    #: str: Name of the method.
    method_name = "Activation function and tabular self-attention"
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

        self.attention_reduction_ratio = 16
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

        # check that the output dimensions of the modality layers are the same
        mod1_output_dim = list(self.mod1_layers.values())[-1][0].out_features
        mod2_output_dim = list(self.mod2_layers.values())[-1][0].out_features
        if mod1_output_dim != mod2_output_dim:
            raise UserWarning(
                "The number of output features of mod1_layers and mod2_layers must be the same for ActivationandSelfAttention. Please change the final layers in the modality layers to have the same number of output features as each other.")

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

        num_channels = x_tab2.size(1)

        # Channel attention
        channel_attention = ChannelAttentionModule(num_features=num_channels,
                                                   reduction_ratio=self.attention_reduction_ratio)
        x_tab2 = channel_attention(x_tab2)

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


class ChannelAttentionModule(nn.Module):
    """
    Channel attention module.

    Attributes
    ----------
    fc1 : nn.Linear
        First fully connected layer.
    relu : nn.ReLU
        ReLU activation function.
    fc2 : nn.Linear
        Second fully connected layer.
    sigmoid : nn.Sigmoid
        Sigmoid activation function.
    """

    def __init__(self, num_features, reduction_ratio=16):
        """

        Parameters
        ----------
        num_features: int
            Number of features of the input tensor.
        reduction_ratio: int
            Reduction ratio of the channel attention module.
        """
        super(ChannelAttentionModule, self).__init__()

        if num_features // reduction_ratio < 1:
            raise UserWarning(
                "first tabular modality dimensions // attention_reduction_ratio < 1. This will cause an error in the model.")

        self.fc1 = nn.Linear(num_features, num_features // reduction_ratio, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_features // reduction_ratio, num_features, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the channel attention module.

        Parameters
        ----------
        x: torch.tensor
            Input tensor.

        Returns
        -------
        torch.tensor
            Output tensor after applying the channel attention module.

        """
        y = self.fc1(x)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
