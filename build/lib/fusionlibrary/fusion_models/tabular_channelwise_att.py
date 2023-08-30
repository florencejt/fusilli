"""
Channel-wise multiplication fusion model for tabular data.
"""

import torch.nn as nn
from fusionlibrary.fusion_models.base_pl_model import ParentFusionModel


class TabularChannelWiseMultiAttention(ParentFusionModel, nn.Module):
    """

    This class implements a model that fuses the two types of tabular data using a
      channel-wise multiplication approach.

    Inspired by the work of Duanmu et al. (2020) [1]: here we use two types of tabular data as
    the multi-modal data instead of image and non-image like in the paper.


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

    Methods
    -------
    forward(x)
      Forward pass of the model.

    References
    ----------
    Duanmu, H., Huang, P. B., Brahmavar, S., Lin, S., Ren, T., Kong, J., Wang, F.,
      & Duong, T. Q. (2020).
    Prediction of Pathological Complete Response to Neoadjuvant Chemotherapy in Breast
      Cancer Using Deep
    Learning with Integrative Imaging, Molecular and Demographic Data. In A. L. Martel,
      P. Abolmaesumi,
    D. Stoyanov, D. Mateus, M. A. Zuluaga, S. K. Zhou, D. Racoceanu, & L. Joskowicz (Eds.),
    Medical Image Computing and Computer Assisted Intervention – MICCAI 2020 (pp. 242–252).
    Springer International Publishing. https://doi.org/10.1007/978-3-030-59713-9_24

    Accompanying code: (our model is inspired by the work of Duanmu et al. (2020) [1])
    https://github.com/HongyiDuanmu26/Prediction-of-pCR-with-Integrative-Deep-Learning/
    blob/main/CustomNet.py

    """

    method_name = "Channel-wise multiplication net (tabular)"
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

        self.set_mod1_layers()
        self.set_mod2_layers()

        # self.fused_dim = list(self.mod1_layers.values())[-1][0].out_features
        # self.set_fused_layers(self.fused_dim)
        # self.set_final_pred_layers()

    def calc_fused_layers(self):
        self.fused_dim = list(self.mod1_layers.values())[-1][0].out_features
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
        x_tab1 = x[0]
        x_tab2 = x[1]

        for i, (k, layer) in enumerate(self.mod1_layers.items()):
            x_tab1 = layer(x_tab1)
            x_tab2 = self.mod2_layers[k](x_tab2)

            x_tab2 = x_tab2 * x_tab1

        out_fuse = x_tab2

        out_fuse = self.fused_layers(out_fuse)

        out = self.final_prediction(out_fuse)

        return [
            out,
        ]
