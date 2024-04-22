"""
Crossmodal multi-head attention for tabular data.
"""

# TODO make 3-tabular data work


import torch.nn as nn
from fusilli.fusionmodels.base_model import ParentFusionModel
import torch
from fusilli.utils import check_model_validity


class TabularCrossmodalMultiheadAttention(ParentFusionModel, nn.Module):
    """Tabular Crossmodal multi-head attention model.

    This class implements a model that fuses the two types of tabular data using a
    cross-modal multi-head attention approach.

    Inspired by the work of Golovanevsky et al. (2021) [1]: here we use two types of tabular data as
    the multi-modal data instead of 3 types in the paper.

    References
    ----------
    Golovanevsky, M., Eickhoff, C., & Singh, R. (2022). Multimodal attention-based
    deep learning for Alzheimer’s disease diagnosis.
    Journal of the American Medical Informatics Association, 29(12),
    2014–2022. https://doi.org/10.1093/jamia/ocac168

    Accompanying code: (our model is inspired by the work of Golovanevsky et al. (2021) [1])
    https://github.com/rsinghlab/MADDi/blob/main/training/train_all_modalities.py

    Attributes
    ----------
    prediction_task : str
        Type of prediction to be performed.
    attention_embed_dim : int
        Number of features of the multihead attention layer.
    mod1_layers : nn.ModuleDict
        Dictionary containing the layers of the first modality.
    mod2_layers : nn.ModuleDict
        Dictionary containing the layers of the second modality.
    fused_dim : int
        Number of features of the fused layers. This is the flattened output size of the
        first tabular modality's layers.
    attention : nn.MultiheadAttention
        Multihead attention layer. Takes in attention_embed_dim features as input.
    tab1_to_embed_dim : nn.Linear
        Linear layer. Takes in fused_dim features as input. This is the input of the
        multihead attention layer.
    tab2_to_embed_dim : nn.Linear
        Linear layer. Takes in fused_dim features as input. This is the input of the
        multihead attention layer.
    relu : nn.ReLU
        ReLU activation function.
    final_prediction : nn.Sequential
        Sequential layer containing the final prediction layers.

    """

    #: str: Name of the method.
    method_name = "Tabular Crossmodal multi-head attention"
    #: str: Type of modality.
    modality_type = "tabular_tabular"
    #: str: Type of fusion.
    fusion_type = "attention"
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
        self.attention_embed_dim = 50

        self.set_mod1_layers()
        self.set_mod2_layers()
        if self.data_dims["mod3_dim"] is not None:
            self.set_mod3_layers()

        self.calc_fused_layers()

        self.relu = nn.ReLU()

    def calc_fused_layers(self):
        """
        Calculate the fused layers.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the number of layers in the two modalities is not the same.
        ValueError
            If dtype of the layers is not nn.ModuleDict.

        """
        check_model_validity.check_dtype(self.mod1_layers, nn.ModuleDict, "mod1_layers")
        check_model_validity.check_dtype(self.mod2_layers, nn.ModuleDict, "mod2_layers")

        # if we've got a 3rd tabular modality
        if self.data_dims["mod3_dim"] is not None:

            # check the layers are valid (if they've been changed)
            check_model_validity.check_dtype(
                self.mod3_layers, nn.ModuleDict, "mod3_layers"
            )

            # check that the number of layers in each modality is the same
            if (
                len(self.mod1_layers) != len(self.mod2_layers)
                or len(self.mod1_layers) != len(self.mod3_layers)
                or len(self.mod2_layers) != len(self.mod3_layers)
            ):
                raise ValueError(
                    "The number of layers in the three modalities must be the same."
                )
        else:
            # check that the number of layers in each modality is the same if we don't have a 3rd modality
            if len(self.mod1_layers) != len(self.mod2_layers):
                raise ValueError(
                    "The number of layers in the two modalities must be the same."
                )

        # Output size of the first tabular modality
        self.fused_dim = list(self.mod1_layers.values())[-1][0].out_features

        mod1_output_size = list(self.mod1_layers.values())[-1][0].out_features
        mod2_output_size = list(self.mod2_layers.values())[-1][0].out_features

        self.tab1_to_embed_dim = nn.Linear(mod1_output_size, self.attention_embed_dim)
        self.tab2_to_embed_dim = nn.Linear(mod2_output_size, self.attention_embed_dim)

        if self.data_dims["mod3_dim"] is not None:
            mod3_output_size = list(self.mod3_layers.values())[-1][0].out_features
            self.tab3_to_embed_dim = nn.Linear(
                mod3_output_size, self.attention_embed_dim
            )

        self.attention = nn.MultiheadAttention(
            embed_dim=self.attention_embed_dim, num_heads=2
        )

        if self.data_dims["mod3_dim"] is None:
            self.set_final_pred_layers(self.attention_embed_dim * 2)
        else:
            self.set_final_pred_layers(self.attention_embed_dim * 6)

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
            Output tensor.
        """

        # ~~ Checks ~~
        check_model_validity.check_model_input(x1)
        check_model_validity.check_model_input(x2)
        if x3 is not None:
            check_model_validity.check_model_input(x3)

        # First step is to pass the tabular data through the layers

        for i, (k, layer) in enumerate(self.mod1_layers.items()):
            x1 = layer(x1)
            x2 = self.mod2_layers[k](x2)
            if x3 is not None:
                x3 = self.mod3_layers[k](x3)

        # Self attention
        out_tab1 = self.tab1_to_embed_dim(x1)
        out_tab1 = self.relu(out_tab1)

        out_tab2 = self.tab2_to_embed_dim(x2)
        out_tab2 = self.relu(out_tab2)

        if x3 is not None:
            out_tab3 = self.tab3_to_embed_dim(x3)
            out_tab3 = self.relu(out_tab3)

        # self attention
        tab2_att = self.attention(out_tab2, out_tab2, out_tab2)[0]
        tab1_att = self.attention(out_tab1, out_tab1, out_tab1)[0]

        if x3 is not None:
            tab3_att = self.attention(out_tab3, out_tab3, out_tab3)[0]

        # cross modal attention between each pair of tabular data

        tab1_tab2_att = self.attention(tab1_att, tab2_att, tab2_att)[0]
        tab2_tab1_att = self.attention(tab2_att, tab1_att, tab1_att)[0]

        crossmodal_att = torch.concat((tab1_tab2_att, tab2_tab1_att), dim=-1)

        if x3 is not None:
            tab1_tab3_att = self.attention(tab1_att, tab3_att, tab3_att)[0]
            tab2_tab3_att = self.attention(tab2_att, tab3_att, tab3_att)[0]
            tab3_tab1_att = self.attention(tab3_att, tab1_att, tab1_att)[0]
            tab3_tab2_att = self.attention(tab3_att, tab2_att, tab2_att)[0]

            crossmodal_att = torch.concat(
                (
                    tab1_tab2_att,
                    tab2_tab1_att,
                    tab1_tab3_att,
                    tab2_tab3_att,
                    tab3_tab1_att,
                    tab3_tab2_att,
                ),
                dim=-1,
            )

        # concatenate
        # merged = torch.concat((crossmodal_att, out_tab1, out_tab2), dim=-1)

        out_fuse = self.final_prediction(crossmodal_att)

        return out_fuse
