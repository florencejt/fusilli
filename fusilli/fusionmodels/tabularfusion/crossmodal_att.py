"""
Crossmodal multi-head attention for tabular data.
"""

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
        self.attention_embed_dim = 50

        self.set_mod1_layers()
        self.set_mod2_layers()
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
        # if mod1 and mod2 have a different number of layers, return error
        check_model_validity.check_dtype(self.mod1_layers, nn.ModuleDict, "mod1_layers")
        check_model_validity.check_dtype(self.mod2_layers, nn.ModuleDict, "mod2_layers")

        if len(self.mod1_layers) != len(self.mod2_layers):
            raise ValueError(
                "The number of layers in the two modalities must be the same."
            )

        self.fused_dim = list(self.mod1_layers.values())[-1][0].out_features

        # self.set_fused_layers(self.fused_dim)

        self.tab1_to_embed_dim = nn.Linear(self.fused_dim, self.attention_embed_dim)
        self.tab2_to_embed_dim = nn.Linear(
            list(self.mod2_layers.values())[-1][0].out_features,
            self.attention_embed_dim,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=self.attention_embed_dim, num_heads=2
        )

        self.set_final_pred_layers(self.attention_embed_dim * 4)

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

        for i, (k, layer) in enumerate(self.mod1_layers.items()):
            x_tab1 = layer(x_tab1)
            x_tab2 = self.mod2_layers[k](x_tab2)

        out_tab2 = self.tab2_to_embed_dim(x_tab2)
        out_tab2 = self.relu(out_tab2)

        out_tab1 = self.tab1_to_embed_dim(x_tab1)
        out_tab1 = self.relu(out_tab1)

        # self attention
        tab2_att = self.attention(out_tab2, out_tab2, out_tab2)[0]
        tab1_att = self.attention(out_tab1, out_tab1, out_tab1)[0]

        # cross modal attention
        tab1_tab2_att = self.attention(tab1_att, tab2_att, tab2_att)[0]
        tab2_tab1_att = self.attention(tab2_att, tab1_att, tab1_att)[0]

        crossmodal_att = torch.concat((tab1_tab2_att, tab2_tab1_att), dim=-1)

        # concatenate
        merged = torch.concat((crossmodal_att, out_tab1, out_tab2), dim=-1)

        out_fuse = self.final_prediction(merged)

        return [
            out_fuse,
        ]
