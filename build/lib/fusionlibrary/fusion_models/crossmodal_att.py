"""
Crossmodal multi-head attention model. This model uses the self attention and cross modal attention
between the two modalities: tabular and image.
"""

import torch.nn as nn
from fusionlibrary.fusion_models.base_pl_model import ParentFusionModel
import torch


class CrossmodalMultiheadAttention(ParentFusionModel, nn.Module):
    """
    Crossmodal multi-head attention model. This model uses the self attention and cross modal attention
    between the two modalities: tabular and image.

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
    attention : nn.MultiheadAttention
        Multihead attention layer. Takes in 50 features as input.
    img_dense : nn.Linear
        Linear layer. Takes in 50 features as input. This is the output of the multihead attention layer.
    to50 : nn.Linear
        Linear layer. Takes in the number of features of the fused layers as input.
    relu : nn.ReLU
        ReLU activation function.
    tab1drops : list
        List containing the dropout layers for the tabular modality.
    final_prediction : nn.Sequential
        Sequential layer containing the final prediction layers. The final prediction layers take in 200 features.

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

    https://github.com/rsinghlab/MADDi/blob/main/training/train_all_modalities.py

    """

    method_name = "Crossmodal multi-head attention"
    modality_type = "tab_img"
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
        """
        ParentFusionModel.__init__(self, pred_type, data_dims, params)

        self.pred_type = pred_type

        self.set_mod1_layers()
        self.set_img_layers()

        self.fused_dim = list(self.img_layers.values())[-1][0].out_channels
        self.set_fused_layers(self.fused_dim)

        self.attention = nn.MultiheadAttention(embed_dim=50, num_heads=2)

        self.img_dense = nn.Linear(50, 1)
        self.to50 = nn.Linear(self.fused_dim, 50)

        self.relu = nn.ReLU()

        self.tab1drops = [nn.Dropout(p=0.5), nn.Dropout(p=0.3), nn.Dropout(p=0.2)]

        self.set_final_pred_layers(200)

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

        for i, (k, layer) in enumerate(self.mod1_layers.items()):
            x_tab1 = layer(x_tab1)
            x_img = self.img_layers[k](x_img)

        out_img = x_img.view(x_img.size(0), -1)
        out_img = self.to50(out_img)
        out_img = self.relu(out_img)

        out_tab1 = x_tab1.view(x_tab1.size(0), -1)
        out_tab1 = self.to50(out_tab1)

        # self attention
        img_att = self.attention(out_img, out_img, out_img)[0]
        tab1_att = self.attention(out_tab1, out_tab1, out_tab1)[0]

        # cross modal attention
        tab1_img_att = self.attention(tab1_att, img_att, img_att)[0]
        img_tab1_att = self.attention(img_att, tab1_att, tab1_att)[0]

        crossmodal_att = torch.concat((tab1_img_att, img_tab1_att), dim=1)

        # concatenate
        merged = torch.concat((crossmodal_att, out_tab1, out_img), dim=1)

        out_fuse = self.final_prediction(merged)

        return [
            out_fuse,
        ]
