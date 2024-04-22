"""
Using activation functions to fuse tabular data, with self-attention on the second tabular modality.
"""

# TODO make 3-tabular data work

# TODO add argument to say which tabular modality is the best one


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
    main_modality: int
        Which modality is the modality that has its feature maps concatenated with the fused feature map? 1, 2, or 3 depending on which modality is the main one.
    attention_modality: int
        Which modality is the attention modality? 1, 2, or 3 depending on which modality has the self-attention function applied to it.
    mod3_layers : nn.ModuleDict (optional)
        Dictionary containing the layers of the 3rd type of tabular data. If 3 tabular data are not provided, this is not used.
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

        self.attention_reduction_ratio = 16
        self.set_mod1_layers()
        self.set_mod2_layers()
        if self.data_dims["mod3_dim"] is not None:
            self.set_mod3_layers()

        self.main_modality = 1  # Which modality has its feature maps concatenated with the fused feature map

        self.attention_modality = (
            2  # Which modality has the self-attention function applied to it
        )

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
        if self.data_dims["mod3_dim"] is not None:
            mod3_output_dim = list(self.mod3_layers.values())[-1][0].out_features
        # New fused dimension is the sum of the output feature map dimensions of one of the modalities (doesn't matter which one because they are the same)
        # And adding the feature map dimension of the "main" modality, again, doesn't matter which one because they are the same
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
                "The number of output features of the modality layers must be the same for Activation fusion. Please change the final layers in the modality layers to have the same number of output features as each other."
            )

        # And if there are 3 modalities, check the third one
        if self.data_dims["mod3_dim"] is not None:
            check_model_validity.check_dtype(
                self.mod3_layers, nn.ModuleDict, "mod3_layers"
            )
            mod3_output_dim = list(self.mod3_layers.values())[-1][0].out_features
            if mod1_output_dim != mod3_output_dim or mod2_output_dim != mod3_output_dim:
                raise UserWarning(
                    "The number of output features of the modality layers must be the same for Activation fusion. Please change the final layers in the modality layers to have the same number of output features as each other."
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
            Input tensor of the first modality.
        x2 : torch.Tensor
            Input tensor of the second modality.
        x3 : torch.Tensor or None
            Input tensor of the third modality. Default is None.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """

        # ~~ Checks ~~
        check_model_validity.check_model_input(x1)
        check_model_validity.check_model_input(x2)

        x_tab1 = x1
        x_tab2 = x2

        inputs = [x_tab1, x_tab2]
        if x3 is not None:
            check_model_validity.check_model_input(x3)
            x_tab3 = x3
            inputs.append(x_tab3)

        # Get the number of channels of the main modality
        num_channels = inputs[self.attention_modality - 1].size(1)
        # print("Num channels", num_channels)

        # Channel attention on the main modality
        channel_attention = ChannelAttentionModule(
            num_features=num_channels, reduction_ratio=self.attention_reduction_ratio
        )
        attention_out = channel_attention(inputs[self.attention_modality - 1])

        if self.attention_modality == 1:
            x_tab1 = attention_out
        for layer in self.mod1_layers.values():
            x_tab1 = layer(x_tab1)

        if self.attention_modality == 2:
            x_tab2 = attention_out
        for layer in self.mod2_layers.values():
            x_tab2 = layer(x_tab2)

        if x3 is not None:
            if self.attention_modality == 3:
                x_tab3 = attention_out
            for layer in self.mod3_layers.values():
                x_tab3 = layer(x_tab3)
            x_tab3 = torch.squeeze(x_tab3, 1)

        x_tab1 = torch.squeeze(x_tab1, 1)
        x_tab2 = torch.squeeze(x_tab2, 1)

        out_fuse = torch.mul(x_tab1, x_tab2)

        if x3 is not None:
            out_fuse = torch.mul(out_fuse, x_tab3)

        outputs = [x_tab1, x_tab2]
        if x3 is not None:
            outputs.append(x_tab3)

        out_fuse = torch.tanh(out_fuse)
        out_fuse = torch.sigmoid(out_fuse)

        # Concatenate the output of the
        out_fuse = torch.cat((out_fuse, outputs[self.main_modality - 1]), dim=1)

        out_fuse = self.fused_layers(out_fuse)

        out = self.final_prediction(out_fuse)

        return out


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
                "Modality dimensions // attention_reduction_ratio < 1. This will cause an error in the model."
            )

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
