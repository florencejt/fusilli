"""
Concat image latent space with tabular data, trained altogether with a custom loss function: MSE + BCE.
"""

import torch.nn as nn
from fusionlibrary.fusion_models.base_pl_model import ParentFusionModel
import torch
from torch.autograd import Variable
import copy


class ConcatImgLatentTabDoubleLoss(ParentFusionModel, nn.Module):
    """
    Concatenating image latent space with tabular data, trained altogether with a custom loss
    function: MSE + BCE.

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
    img_layers : dict
        Dictionary containing the layers of the image data.
    fused_layers : nn.Sequential
        Sequential layer containing the fused layers.
    final_prediction : nn.Sequential
        Sequential layer containing the final prediction layers. The final prediction layers
            take in the number of features of the fused layers as input.
    custom_loss : nn.Module
        Additional loss function to be used for training the model.
    latent_dim : int
        Size of the latent space.
    encoder : nn.Sequential
        Sequential layer containing the encoder layers.
    decoder : nn.Sequential
        Sequential layer containing the decoder layers.
    fc1 : nn.Linear
        Linear layer to be used for the latent space.

    Methods
    -------
    forward(x)
        Forward pass of the model.

    """

    method_name = (
        "Concatenating image latent space with tabular data, trained altogether"
    )
    modality_type = "tab_img"
    fusion_type = "subspace"

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
        self.custom_loss = nn.MSELoss()
        self.img_dims = data_dims[-1]

        # self.set_img_layers()
        self.latent_dim = 256  # You can adjust the latent space size

        # self.fused_dim = (
        #     self.mod1_dim + list(self.img_layers.values())[-1][0].out_channels
        # )
        # self.set_fused_layers(self.fused_dim)
        # self.set_final_pred_layers()

        if len(self.img_dims) == 2:  # 2D images
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

        elif len(self.img_dims) == 3:
            self.encoder = nn.Sequential(
                nn.Conv3d(1, 16, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Conv3d(16, 32, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Conv3d(32, self.latent_dim, kernel_size=3, stride=1),
                nn.ReLU(),
                # nn.Conv3d(128, 256, kernel_size=3, stride=1),
                # nn.ReLU(),
            )

            self.decoder = nn.Sequential(
                # nn.ConvTranspose3d(256, 128, kernel_size=3, stride=1, output_padding=1),
                # nn.ReLU(),
                nn.ConvTranspose3d(self.latent_dim, 32, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.ConvTranspose3d(32, 16, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.ConvTranspose3d(16, 1, kernel_size=3, stride=1),
                nn.Sigmoid(),
                nn.Upsample(size=self.img_dims, mode="trilinear", align_corners=False),
            )
        else:
            raise ValueError("Invalid image dimensions.")

        self.calc_fused_layers()

    def calc_fused_layers(self):
        # get dummy conv output
        self.flatten = nn.Flatten()

        # size of final encoder output
        dummy_conv_output = Variable(torch.rand((1,) + tuple(self.data_dims[-1])))
        dummy_conv_output = self.encoder(dummy_conv_output)
        n_size = dummy_conv_output.data.view(1, -1).size(1)

        # make linear layer to reduce to latent dim
        self.linear_to_lat_dim = nn.Linear(n_size, self.latent_dim)

        # add extra layers to encoder
        self.new_encoder = copy.deepcopy(self.encoder)
        self.new_encoder.append(self.flatten)
        self.new_encoder.append(self.linear_to_lat_dim)

        # add extra layer to decoder to get right shape for first decoding layer
        self.new_decoder = copy.deepcopy(self.decoder)

        self.linear_to_decoder = nn.Linear(
            self.latent_dim, self.new_decoder[0].in_channels
        )
        # TODO make this work for 3d images too
        self.unflatten = nn.Unflatten(1, (self.new_decoder[0].in_channels, 1, 1))

        self.new_decoder.insert(0, self.linear_to_decoder)
        self.new_decoder.insert(1, self.unflatten)

        self.new_decoder.append(nn.Sigmoid()),  # Output is scaled between 0 and 1
        self.new_decoder.append(
            nn.Upsample(size=self.img_dims, mode="bilinear", align_corners=False)
        )

        self.fused_dim = self.latent_dim + self.data_dims[0]
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
        list : list
            List containing the output data: prediction and reconstructed image.
            [ [prediction], [reconstructed_image] ]
        """
        x_tab = x[0].squeeze(dim=1)
        x_img = x[1].unsqueeze(dim=1)

        # encoder
        encoded_img = self.new_encoder(x_img)
        # for i, layer in enumerate(self.enc_layers.values()):
        #     x_img = layer(x_img)

        # latent space
        latent_space = encoded_img

        # encoded_img = encoded_img.view(encoded_img.size(0), -1)  # Flatten
        # latent_space = nn.ReLU()(self.fc1(encoded_img))

        # decoder
        print(self.new_decoder)
        reconstructed_image = self.new_decoder(encoded_img)
        # for i, layer in enumerate(self.dec_layers.values()):
        #     decoder_input = layer(decoder_input)

        reconstructed_image = torch.sigmoid(reconstructed_image)

        # concatenate latent space with tabular data
        fused_data = torch.cat([latent_space, x_tab], dim=1)

        # put fused data through some joint layers
        out_fuse = self.fused_layers(fused_data)

        # final prediction
        out = self.final_prediction(out_fuse)

        return [out, reconstructed_image.squeeze(dim=1)]
