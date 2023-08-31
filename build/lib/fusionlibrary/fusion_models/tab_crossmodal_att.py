"""
Crossmodal multi-head attention for tabular data.
"""

import torch.nn as nn
from fusionlibrary.fusion_models.base_pl_model import ParentFusionModel
import torch


class TabularCrossmodalMultiheadAttention(ParentFusionModel, nn.Module):
    """

    This class implements a model that fuses the two types of tabular data using a
        cross-modal multi-head attention approach.

    Inspired by the work of Golovanevsky et al. (2021) [1]: here we use two types of tabular data as
    the multi-modal data instead of 3 types in the paper.

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
    mod2_layers : dict
        Dictionary containing the layers of the 2nd type of tabular data.
    fused_layers : nn.Sequential
        Sequential layer containing the fused layers.
    final_prediction : nn.Sequential
        Sequential layer containing the final prediction layers.
    fused_dim : int
        Dimension of the fused layer.
    attention : nn.MultiheadAttention
        Multi-head attention layer.
    to50 : nn.Linear
        Linear layer to reduce the dimension of the data to 50.
    relu : nn.ReLU
        ReLU activation function.
    clindrops : list
        List containing the dropout layers.

    Methods
    -------
    forward(x)
        Forward pass of the model.

    References
    ----------
    Golovanevsky, M., Eickhoff, C., & Singh, R. (2022). Multimodal attention-based
    deep learning for Alzheimer’s disease diagnosis.
    Journal of the American Medical Informatics Association, 29(12),
    2014–2022. https://doi.org/10.1093/jamia/ocac168

    Accompanying code: (our model is inspired by the work of Golovanevsky et al. (2021) [1])
    https://github.com/rsinghlab/MADDi/blob/main/training/train_all_modalities.py

    """

    method_name = "Tabular Crossmodal multi-head attention"
    modality_type = "both_tab"
    fusion_type = "attention"

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
        self.attention_embed_dim = 50

        self.set_mod1_layers()
        self.set_mod2_layers()
        self.calc_fused_layers()

        # self.fused_dim = list(self.mod1_layers.values())[-1][0].out_features
        # self.set_fused_layers(self.fused_dim)

        # self.attention = nn.MultiheadAttention(embed_dim=self.attention_embed_dim, num_heads=2)

        # self.to50 = nn.Linear(self.fused_dim, self.attention_embed_dim)

        self.relu = nn.ReLU()

        self.clindrops = [nn.Dropout(p=0.5), nn.Dropout(p=0.3), nn.Dropout(p=0.2)]

        # self.set_final_pred_layers(200)

    def calc_fused_layers(self):
        """
        Calculate the fused layers.
        """
        # if mod1 and mod2 have a different number of layers, return error
        if len(self.mod1_layers) != len(self.mod2_layers):
            raise ValueError(
                "The number of layers in the two modalities must be the same."
            )

        self.fused_dim = list(self.mod1_layers.values())[-1][0].out_features

        self.set_fused_layers(self.fused_dim)

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
        x : list
            List containing the input data.

        Returns
        -------
        list
            List containing the output of the model.
        """
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
