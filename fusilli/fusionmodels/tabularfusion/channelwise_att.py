"""
Channel-wise multiplication fusion model for tabular data.
"""

import torch.nn as nn
from fusilli.fusionmodels.base_model import ParentFusionModel
from fusilli.utils import check_model_validity


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
    https://github.com/HongyiDuanmu26/Prediction-of-pCR-with-Integrative-Deep-Learning/blob/main/CustomNet.py


    Attributes
    ----------
    mod1_layers : nn.ModuleDict
      Dictionary containing the layers of the 1st type of tabular data.
    mod2_layers : nn.ModuleDict
      Dictionary containing the layers of the 2nd type of tabular data.
    mod3_layers : nn.ModuleDict (optional)
        Dictionary containing the layers of the 3rd type of tabular data. If 3 tabular data are not provided, this is not used.
    match_dim_layers: dict
        Dictionary containing module dictionaries of linear layers to make the dimensions of the types of tabular data the same at each layer.
        This is done to ensure that the channel-wise multiplication can be performed and does not alter the original layers.
    fused_dim : int
      Number of features of the fused layers. This is the output size of the
      final main tabular data's layers.
    fused_layers : nn.Sequential
      Sequential layer containing the fused layers.
    final_prediction : nn.Sequential
      Sequential layer containing the final prediction layers.

    """

    #: str: Name of the method.
    method_name = "Channel-wise multiplication net (tabular)"
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
        data_dims : list
          List containing the dimensions of the data.
        multiclass_dimensions : int
                Number of classes in the dataset.
        """
        ParentFusionModel.__init__(
            self, prediction_task, data_dims, multiclass_dimensions
        )

        self.prediction_task = prediction_task

        self.set_mod1_layers()
        self.set_mod2_layers()
        if self.data_dims["mod3_dim"] is not None:
            self.set_mod3_layers()

        # Which modality is the main one?
        self.main_modality = 1  # 1, 2, or 3 depending on which modality is the main one

        self.get_fused_dim()
        self.set_fused_layers(self.fused_dim)

        self.calc_fused_layers()

    def get_fused_dim(self):
        """
        Returns the number of features of the fused layers.

        Returns
        -------
        None.
        """

        if self.main_modality == 1:
            self.fused_dim = list(self.mod1_layers.values())[-1][0].out_features
        elif self.main_modality == 2:
            self.fused_dim = list(self.mod2_layers.values())[-1][0].out_features
        elif self.main_modality == 3:
            self.fused_dim = list(self.mod3_layers.values())[-1][0].out_features
        else:
            raise ValueError("main_modality must be 1, 2, or 3")

        # self.fused_dim = list(self.mod2_layers.values())[-1][0].out_features

    def calc_fused_layers(self):
        """
        Calculates the fusion layers and creates layer groups to match the dimensions of the tabular data.

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

        # get the fused dimension from the output of the last layer of the main modality
        self.get_fused_dim()
        # check the fused layers are valid
        self.fused_layers, out_dim = check_model_validity.check_fused_layers(
            self.fused_layers, self.fused_dim
        )
        # set the final prediction layers
        self.set_final_pred_layers(out_dim)

        ############################################################
        # Creating extra layers to match the dimensions of the tabular data at each method step
        ############################################################

        self.main_mod_layers = getattr(self, f"mod{self.main_modality}_layers")
        self.match_dim_layers = {
            "1_to_2": nn.ModuleDict(),
            "1_to_3": nn.ModuleDict(),
            "2_to_1": nn.ModuleDict(),
            "2_to_3": nn.ModuleDict(),
            "3_to_1": nn.ModuleDict(),
            "3_to_2": nn.ModuleDict(),
        }

        for key in self.mod1_layers.keys():
            # get output sizes of each modality
            layer1 = self.mod1_layers[key]
            layer2 = self.mod2_layers[key]
            layer3 = (
                self.mod3_layers[key]
                if self.data_dims["mod3_dim"] is not None
                else None
            )

            layer1_out = layer1[0].out_features
            layer2_out = layer2[0].out_features
            layer3_out = layer3[0].out_features if layer3 is not None else None

            # check if the output sizes are different and create linear layer if needed
            if layer1_out != layer2_out:
                self.match_dim_layers["1_to_2"][key] = nn.Linear(layer1_out, layer2_out)
                self.match_dim_layers["2_to_1"][key] = nn.Linear(layer2_out, layer1_out)
            else:
                self.match_dim_layers["1_to_2"][key] = nn.Identity()
                self.match_dim_layers["2_to_1"][key] = nn.Identity()

            if self.data_dims["mod3_dim"] is not None:
                if layer1_out != layer3_out:
                    self.match_dim_layers["1_to_3"][key] = nn.Linear(
                        layer1_out, layer3_out
                    )
                    self.match_dim_layers["3_to_1"][key] = nn.Linear(
                        layer3_out, layer1_out
                    )
                else:
                    self.match_dim_layers["1_to_3"][key] = nn.Identity()
                    self.match_dim_layers["3_to_1"][key] = nn.Identity()

                if layer2_out != layer3_out:
                    self.match_dim_layers["2_to_3"][key] = nn.Linear(
                        layer2_out, layer3_out
                    )
                    self.match_dim_layers["3_to_2"][key] = nn.Linear(
                        layer3_out, layer2_out
                    )
                else:
                    self.match_dim_layers["2_to_3"][key] = nn.Identity()
                    self.match_dim_layers["3_to_2"][key] = nn.Identity()

    def forward(self, x1, x2, x3=None):
        """
        Forward pass of the model.

        Parameters
        ----------
        x1 : torch.Tensor
            1st modality of tabular data.
        x2 : torch.Tensor
            2nd modality of tabular data.
        x3 : torch.Tensor
            Input tensor for the third modality. Default is None.

        Returns
        -------
        torch.Tensor
            Output tensor of the model.
        """

        # ~~ Checks ~~
        check_model_validity.check_model_input(x1)
        check_model_validity.check_model_input(x2)
        if x3 is not None:
            check_model_validity.check_model_input(x3)

        # which input is the main one?
        if self.main_modality == 1:
            x_main = x1

            for i, (k, layer) in enumerate(self.mod1_layers.items()):
                x_main = layer(x_main)

                x2 = self.mod2_layers[k](x2)
                # layer to get the tab2 feature maps to be the same size as tab1
                new_x2 = self.match_dim_layers["2_to_1"][k](x2)
                x_main = x_main * new_x2

                if x3 is not None:
                    x3 = self.mod3_layers[k](x3)
                    # layer to get the tab3 feature maps to be the same size as tab1
                    new_x3 = self.match_dim_layers["3_to_1"][k](x3)
                    x_main = x_main * new_x3

        elif self.main_modality == 2:
            x_main = x2

            for i, (k, layer) in enumerate(self.mod2_layers.items()):
                x_main = layer(x_main)

                x1 = self.mod1_layers[k](x1)
                # layer to get the tab1 feature maps to be the same size as tab2
                new_x1 = self.match_dim_layers["1_to_2"][k](x1)
                x_main = x_main * new_x1

                if x3 is not None:
                    x3 = self.mod3_layers[k](x3)
                    # layer to get the tab3 feature maps to be the same size as tab2
                    new_x3 = self.match_dim_layers["3_to_2"][k](x3)
                    x_main = x_main * new_x3

        elif self.main_modality == 3:
            x_main = x3

            for i, (k, layer) in enumerate(self.mod3_layers.items()):
                x_main = layer(x_main)

                x1 = self.mod1_layers[k](x1)
                x2 = self.mod2_layers[k](x2)

                # layer to get the tab1 feature maps to be the same size as tab3
                new_x1 = self.match_dim_layers["1_to_3"][k](x1)
                x_main = x_main * new_x1

                # layer to get the tab2 feature maps to be the same size as tab3
                new_x2 = self.match_dim_layers["2_to_3"][k](x2)
                x_main = x_main * new_x2

        out_fuse = x_main

        out_fuse = self.fused_layers(out_fuse)

        out = self.final_prediction(out_fuse)

        return out
