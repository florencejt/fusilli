import torch.nn as nn
from fusionlibrary.fusion_models.base_pl_model import ParentFusionModel


class ImageChannelWiseMultiAttention(ParentFusionModel, nn.Module):
    """
    Channel-wise multiplication net (MRI)

    Inspired by the work of Duanmu et al. (2020) [1]., we use channel-wise multiplication to combine
    tabular data and image data.

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
    img_layers : dict
        Dictionary containing the layers of the image data.
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

    Duanmu, H., Huang, P. B., Brahmavar, S., Lin, S., Ren, T., Kong, J.,
    Wang, F., & Duong, T. Q. (2020).
    Prediction of Pathological Complete Response to Neoadjuvant Chemotherapy in Breast
    Cancer Using Deep
    Learning with Integrative Imaging, Molecular and Demographic Data. In A.
      L. Martel, P. Abolmaesumi,
    D. Stoyanov, D. Mateus, M. A. Zuluaga, S. K. Zhou, D. Racoceanu, & L. Joskowicz (Eds.),
    Medical Image Computing and Computer Assisted Intervention – MICCAI 2020
    (pp. 242–252).
    Springer International Publishing. https://doi.org/10.1007/978-3-030-59713-9_24

    Accompanying code: (our model is inspired by the work of Duanmu et al. (2020) [1])
    https://github.com/HongyiDuanmu26/Prediction-of-pCR-with-Integrative-Deep-Learning/
    blob/main/CustomNet.py

    """

    method_name = "Channel-wise multiplication net (image)"
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
        self.set_final_pred_layers()

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
        x_tab1 = x[0].squeeze(dim=1)
        x_img = x[1].unsqueeze(dim=1)

        for i, (k, layer) in enumerate(self.mod1_layers.items()):
            x_tab1 = layer(x_tab1)
            x_img = self.img_layers[k](x_img)

            x_img = x_img * x_tab1[:, :, None, None, None]

        out_fuse = x_img.view(x_img.size(0), -1)

        out_fuse = self.fused_layers(out_fuse)

        out = self.final_prediction(out_fuse)

        return [
            out,
        ]
