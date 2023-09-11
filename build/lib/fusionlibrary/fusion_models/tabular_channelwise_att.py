"""
Channel-wise multiplication fusion model for tabular data.
"""

import torch.nn as nn
from fusionlibrary.fusion_models.base_pl_model import ParentFusionModel
from fusionlibrary.utils.pl_utils import check_valid_modification_dtype


class TabularChannelWiseMultiAttention(ParentFusionModel, nn.Module):
    """Channel-wise multiplication fusion model for tabular data.

    This class implements a model that fuses the two types of tabular data using a
    channel-wise multiplication approach.

    If the two types of tabular data have different feature dimensions at each layer, the model will
    use a linear layer to make the dimensions the same. This is done to ensure that the
    channel-wise multiplication can be performed.

    Inspired by the work of Duanmu et al. (2020) [1]: here we use two types of tabular data as
    the multi-modal data instead of image and non-image like in the paper.

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


    Attributes
    ----------
    mod1_layers : dict
      Dictionary containing the layers of the 1st type of tabular data.
    mod2_layers : dict
      Dictionary containing the layers of the 2nd type of tabular data.
    match_dim_layers : nn.ModuleDict
      Module dictionary containing the linear layers to make the dimensions of the two types of
      tabular data the same. This is done to ensure that the channel-wise multiplication can be
      performed. This doesn't change the mod1_layers or mod2_layers, it just makes the outputs
      multipliable.
    fused_dim : int
      Number of features of the fused layers. This is the output size of the
      2nd type of tabular data's layers.
    fused_layers : nn.Sequential
      Sequential layer containing the fused layers.
    final_prediction : nn.Sequential
      Sequential layer containing the final prediction layers.

    """

    # str: Name of the method.
    method_name = "Channel-wise multiplication net (tabular)"
    # str: Type of modality.
    modality_type = "both_tab"
    # str: Type of fusion.
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
        self.set_mod2_layers()
        self.calc_fused_layers()

    def calc_fused_layers(self):
        """
        Calculates the fusion layers.

        Returns
        -------
        None

        Raises
        ------
        ValueError
          If the number of layers in the two modalities is different.
        ValueError
          If dtype of the layers is not nn.ModuleDict.
        """
        # if mod1 and mod2 have a different number of layers, return error

        if len(self.mod1_layers) != len(self.mod2_layers):
            raise ValueError(
                "The number of layers in the two modalities must be the same."
            )

        check_valid_modification_dtype(self.mod1_layers, nn.ModuleDict, "mod1_layers")
        check_valid_modification_dtype(self.mod2_layers, nn.ModuleDict, "mod2_layers")

        self.match_dim_layers = nn.ModuleDict()

        # Iterate through your ModuleDict keys
        for key in self.mod1_layers.keys():
            layer_mod1 = self.mod1_layers[key]
            layer_mod2 = self.mod2_layers[key]

            layer_mod1_out = layer_mod1[0].out_features
            layer_mod2_out = layer_mod2[0].out_features

            # Check if the output sizes are different and create linear layer if needed
            if layer_mod1_out != layer_mod2_out:
                self.match_dim_layers[key] = nn.Linear(layer_mod1_out, layer_mod2_out)
            else:
                self.match_dim_layers[key] = nn.Identity()

        self.fused_dim = list(self.mod2_layers.values())[-1][0].out_features
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

            # layer to get the feature maps to be the same size if they have been modified to not be
            if x_tab1.shape[1] != x_tab2.shape[1]:
                # layer to make tab1 output the same size as tab2
                new_x_tab1 = self.match_dim_layers[k](x_tab1)
                x_tab2 = x_tab2 * new_x_tab1
            else:
                x_tab2 = x_tab2 * x_tab1

        out_fuse = x_tab2

        out_fuse = self.fused_layers(out_fuse)

        out = self.final_prediction(out_fuse)

        return [
            out,
        ]
