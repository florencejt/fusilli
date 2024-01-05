"""
Image-channel-wise attention fusion model.
"""
import torch.nn as nn
from fusilli.fusionmodels.base_model import ParentFusionModel
import torch
from torch.autograd import Variable
from fusilli.utils import check_model_validity


class ImageChannelWiseMultiAttention(ParentFusionModel, nn.Module):
    """
    Channel-wise multiplication net with image and tabular

    If the tabular layers and the image layers have different feature maps dimensions at each
    layer, the model will use a linear layer to make the tabular dimensions equal to the image
    layer dimensions. This is done to ensure that the channel-wise multiplication can be performed.

    Inspired by the work of Duanmu et al. (2020) [1]., we use channel-wise multiplication to combine
    tabular data and image data.

    References
    ----------

    Duanmu, H., Huang, P. B., Brahmavar, S., Lin, S., Ren, T., Kong, J.,
    Wang, F., & Duong, T. Q. (2020).
    Prediction of Pathological Complete Response to Neoadjuvant Chemotherapy in Breast
    Cancer Using Deep Learning with Integrative Imaging, Molecular and Demographic Data. In A.
    L. Martel, P. Abolmaesumi,
    D. Stoyanov, D. Mateus, M. A. Zuluaga, S. K. Zhou, D. Racoceanu, & L. Joskowicz (Eds.),
    Medical Image Computing and Computer Assisted Intervention – MICCAI 2020
    (pp. 242–252).
    Springer International Publishing. https://doi.org/10.1007/978-3-030-59713-9_24

    Accompanying code: (our model is inspired by the work of Duanmu et al. (2020) [1])
    https://github.com/HongyiDuanmu26/Prediction-of-pCR-with-Integrative-Deep-Learning/blob/main/CustomNet.py


    Attributes
    ----------
    mod1_layers : nn.ModuleDict
        Dictionary containing the layers of the 1st type of tabular data.
    img_layers : nn.ModuleDict
        Dictionary containing the layers of the image data.
    match_dim_layers : nn.ModuleDict
        Module dictionary containing the linear layers to make the dimensions of the two types of
        data the same. This is done to ensure that the channel-wise multiplication can be performed.
        This doesn't change the mod1_layers or img_layers, it just makes the outputs multipliable.
    fused_dim : int
        Number of features of the fused layers. This is the flattened output size of the
        image layers.
    fused_layers : nn.Sequential
        Sequential layer containing the fused layers.
    final_prediction : nn.Sequential
        Sequential layer containing the final prediction layers.
    """

    #: str: Name of the method.
    method_name = "Channel-wise Image attention"
    #: str: Type of modality.
    modality_type = "tabular_image"
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

        self.set_mod1_layers()
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

        dummy_conv_output = Variable(torch.rand((1,) + tuple(self.img_dim)))
        for layer in self.img_layers.values():
            dummy_conv_output = layer(dummy_conv_output)
        flattened_img_output_size = dummy_conv_output.data.view(1, -1).size(1)

        self.fused_dim = flattened_img_output_size

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
        ValueError
            If the image dimensions are not valid. (Conv2D used for 3D img and vice versa)
        """

        check_model_validity.check_dtype(self.mod1_layers, nn.ModuleDict, "mod1_layers")
        check_model_validity.check_dtype(self.img_layers, nn.ModuleDict, "img_layers")

        check_model_validity.check_img_dim(self.img_layers, self.img_dim, "img_layers")
        # if number of imaging layers does not equal number of tabular layers, return error
        if len(self.mod1_layers) != len(self.img_layers):
            raise ValueError(
                "The number of layers in the two modalities must be the same."
            )

        self.get_fused_dim()
        self.fused_layers, out_dim = check_model_validity.check_fused_layers(
            self.fused_layers, self.fused_dim
        )
        self.set_final_pred_layers(out_dim)

        # Linear layer to make the dimensions of the two types of data the same

        self.match_dim_layers = nn.ModuleDict()

        # Iterate through your ModuleDict keys
        for key in self.mod1_layers.keys():
            layer_mod1 = self.mod1_layers[key]
            layer_mod2 = self.img_layers[key]

            layer_mod1_out = layer_mod1[0].out_features
            layer_mod2_out = layer_mod2[0].out_channels

            # Check if the output sizes are different and create linear layer if needed
            if layer_mod1_out != layer_mod2_out:
                self.match_dim_layers[key] = nn.Linear(layer_mod1_out, layer_mod2_out)
            else:
                self.match_dim_layers[key] = nn.Identity()

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Tensor containing the image data.

        Returns
        -------
        list
            List containing the output of the model.
        """

        # ~~ Checks ~~
        check_model_validity.check_model_input(x)

        x_tab1 = x[0].squeeze(dim=1)
        x_img = x[1]

        for i, (k, layer) in enumerate(self.mod1_layers.items()):
            x_tab1 = layer(x_tab1)
            x_img = self.img_layers[k](x_img)

            # layer to get the feature maps to be the same size if they have been modified to not be
            if x_tab1.shape[1] != x_img.shape[1]:
                new_x_tab1 = self.match_dim_layers[k](x_tab1)
                if len(x_img.shape) == 5:
                    x_img = x_img * new_x_tab1[:, :, None, None, None]
                elif len(x_img.shape) == 4:
                    x_img = x_img * new_x_tab1[:, :, None, None]

            else:
                if len(x_img.shape) == 5:
                    x_img = x_img * x_tab1[:, :, None, None, None]
                elif len(x_img.shape) == 4:
                    x_img = x_img * x_tab1[:, :, None, None]

        out_fuse = x_img.view(x_img.size(0), -1)

        out_fuse = self.fused_layers(out_fuse)

        out = self.final_prediction(out_fuse)

        return [
            out,
        ]
