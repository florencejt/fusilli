"""
Concat image latent space with tabular data, trained altogether with a custom loss function: MSE + BCE.
"""

import torch.nn as nn
from fusilli.fusionmodels.base_model import ParentFusionModel
import torch
from torch.autograd import Variable
import copy
from fusilli.utils import check_model_validity


class ConcatImgLatentTabDoubleLoss(ParentFusionModel, nn.Module):
    """
    Concatenating image latent space with tabular data, trained altogether with a custom loss
    function: MSE + BCE.

    Attributes
    ----------
    prediction_task : str
        Type of prediction to be performed. Binary, regression or multiclass.
    fused_layers : nn.Sequential
        Sequential layer containing the fused layers defined with :func:`calc_fused_layers()`.
    final_prediction : nn.Sequential
        Sequential layer containing the final prediction layers. The final prediction layers
        take in the number of features of the fused layers as input.
    custom_loss : nn.Module
        Additional loss function to be used for training the model. Default is MSELoss.
    latent_dim : int
        Size of the latent space. Default is 256.
    encoder : nn.Sequential
        Sequential layer containing the encoder layers. Default for 2D image is:

        .. code-block:: python

            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

    decoder : nn.Sequential
        Sequential layer containing the decoder layers. Default for 2D image is:

        .. code-block:: python

            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            )

    new_encoder : nn.Sequential
        Sequential layer containing the encoder layers and the additional layers defined with
        :meth:`~ConcatImgLatentTabDoubleLoss.calc_fused_layers()`.
    new_decoder : nn.Sequential
        Sequential layer containing the decoder layers and the additional layers defined with
        :meth:`~ConcatImgLatentTabDoubleLoss.calc_fused_layers()`.
    fused_dim : int
        Size of the fused layers: latent dimension size + tabular data dimension size.
    """

    #: str: Name of the method.
    method_name = (
        "Trained Together Latent Image + Tabular Data"
    )
    #: str: Type of modality.
    modality_type = "tabular_image"
    #: str: Type of fusion.
    fusion_type = "subspace"

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
        self.custom_loss = nn.MSELoss()
        self.img_dim = data_dims[-1]

        self.latent_dim = 256  # You can adjust the latent space size

        if len(self.img_dim) == 2:  # 2D images
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 100x100x1 -> 100x100x32
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 100x100x32 -> 50x50x32
                nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 50x50x32 -> 50x50x64
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 50x50x64 -> 25x25x64
                nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 25x25x64 -> 25x25x128
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 25x25x128 -> 12x12x128
            )

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(
                    128, 64, kernel_size=2, stride=2
                ),  # 12x12x128 -> 25x25x64
                nn.ReLU(),
                nn.ConvTranspose2d(
                    64, 32, kernel_size=2, stride=2
                ),  # 25x25x64 -> 50x50x32
                nn.ReLU(),
                nn.ConvTranspose2d(
                    32, 1, kernel_size=2, stride=2
                ),  # 50x50x32 -> 100x100x1
            )

        elif len(self.img_dim) == 3:
            self.encoder = nn.Sequential(
                nn.Conv3d(1, 16, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=2, stride=2),
                nn.Conv3d(16, 32, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=2, stride=2),
                nn.Conv3d(32, self.latent_dim, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=2, stride=2),
            )

            self.decoder = nn.Sequential(
                # nn.ConvTranspose3d(256, 128, kernel_size=3, stride=1, output_padding=1),
                # nn.ReLU(),
                nn.ConvTranspose3d(self.latent_dim, 32, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.ConvTranspose3d(32, 16, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.ConvTranspose3d(16, 1, kernel_size=3, stride=1),
            )
        else:
            raise ValueError("Invalid image dimensions.")

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

        self.fused_dim = self.latent_dim + self.mod1_dim

    def calc_fused_layers(self):
        """
        Calculate the fused layers. If layer sizes are modified, this function will be called again to adjust the
        fused layers.

        Returns
        -------
        None
        """

        check_model_validity.check_dtype(self.encoder, nn.Sequential, "encoder")
        check_model_validity.check_dtype(self.decoder, nn.Sequential, "decoder")
        check_model_validity.check_dtype(self.latent_dim, int, "latent_dim")

        self.get_fused_dim()

        # check fused layers
        self.fused_layers, out_dim = check_model_validity.check_fused_layers(
            self.fused_layers, self.fused_dim
        )

        if self.latent_dim < 1:
            raise ValueError(
                f"Incorrect attribute range: The latent dimension must be greater than 0. The latent dimension is "
                f"currently: ",
                self.latent_dim,
            )

        check_model_validity.check_img_dim(self.encoder, self.img_dim, "encoder")
        check_model_validity.check_img_dim(self.decoder, self.img_dim, "decoder")

        # size of final encoder output
        dummy_conv_output = Variable(torch.rand((1,) + tuple(self.img_dim)))
        dummy_conv_output = self.encoder(dummy_conv_output)
        n_size = dummy_conv_output.data.view(1, -1).size(1)

        # add extra layers to encoder
        self.new_encoder = copy.deepcopy(self.encoder)
        self.new_encoder.append(nn.Flatten())
        self.new_encoder.append(nn.Linear(n_size, self.latent_dim))

        # add extra layer to decoder to get right shape for first decoding layer
        self.new_decoder = copy.deepcopy(self.decoder)

        first_decoder_layer_inchannels = self.new_decoder[0].in_channels
        self.new_decoder.insert(
            0, nn.Linear(self.latent_dim, first_decoder_layer_inchannels)
        )

        if len(self.img_dim) == 3:
            self.new_decoder.insert(
                1, nn.Unflatten(1, (first_decoder_layer_inchannels, 1, 1, 1))
            )
        elif len(self.img_dim) == 2:
            self.new_decoder.insert(
                1, nn.Unflatten(1, (first_decoder_layer_inchannels, 1, 1))
            )

        self.new_decoder.append(nn.Sigmoid()),  # Output is scaled between 0 and 1

        if len(self.img_dim) == 3:
            self.new_decoder.append(
                nn.Upsample(size=self.img_dim, mode="trilinear", align_corners=False)
            )
        elif len(self.img_dim) == 2:
            self.new_decoder.append(
                nn.Upsample(size=self.img_dim, mode="bilinear", align_corners=False)
            )

        self.set_final_pred_layers(out_dim)

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : tuple
            Tuple containing the input data.

        Returns
        -------
        list : list
            List containing the output data: prediction and reconstructed image.
            [ [prediction], [reconstructed_image] ]
        """

        check_model_validity.check_model_input(x)

        x_tab = x[0].squeeze(dim=1)
        x_img = x[1]

        # encoder
        encoded_img = self.new_encoder(x_img)

        # latent space
        latent_space = encoded_img

        # decoder
        reconstructed_image = self.new_decoder(encoded_img)
        reconstructed_image = torch.sigmoid(reconstructed_image)

        # concatenate latent space with tabular data
        fused_data = torch.cat([latent_space, x_tab], dim=1)

        # put fused data through some joint layers
        out_fuse = self.fused_layers(fused_data)

        # final prediction
        out = self.final_prediction(out_fuse)

        return [out, reconstructed_image]
