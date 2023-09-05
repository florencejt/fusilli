"""
Concatenating the latent img space and tabular data. The latent image space is
trained separately from both the tabular data and the labels, using the
img_latent_subspace_method class.
"""

import torch.nn as nn
from fusionlibrary.fusion_models.base_pl_model import ParentFusionModel
import torch
import pytorch_lightning as pl
from fusionlibrary.utils.pl_utils import init_trainer
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from torch.autograd import Variable
import copy


class ConcatImgLatentTabDoubleTrain(ParentFusionModel, nn.Module):
    """
    Concatenating the latent img space and tabular data. The latent image space is
    trained separately from both the tabular data and the labels, using the
    img_latent_subspace_method class.

    Attributes
    ----------
    pred_type : str
        Type of prediction to be performed.
    subspace_method : class
        Class containing the method to train the latent image space.
        Default is :func:`img_latent_subspace_method`.
    latent_dim : int
        Dimension of the latent image space once we encode it down. Taken from the
        subspace_method class and inferred from the dimensions of the input data to the model.
    enc_img_layer : nn.Linear
        Linear layer to reduce the dimension of the latent image space.
        Calculated with :meth:`~ConcatImgLatentTabDoubleTrain.calc_fused_layers()`.
    fused_dim : int
        Dimension of the fused layers.
        Calculated with :meth:`~ConcatImgLatentTabDoubleTrain.calc_fused_layers()`.
    fused_layers : nn.Sequential
        Sequential layer containing the fused layers.
        Calculated with :meth:`~ConcatImgLatentTabDoubleTrain.calc_fused_layers()`.
    final_prediction : nn.Sequential
        Sequential layer containing the final prediction layers. The final prediction layers
        take in the number of features of the fused layers as input.
        Calculated with :meth:`~ConcatImgLatentTabDoubleTrain.calc_fused_layers()`.

    """

    # str: Name of the method.
    method_name = "Concatenating latent img space and tabular data training separately"
    # str: Type of modality.
    modality_type = "tab_img"
    # str: Type of fusion.
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
        self.subspace_method = concat_img_latent_tab_subspace_method

        self.calc_fused_layers()

    def calc_fused_layers(self):
        """
        Calculate the fused layers. If layer sizes are modified, this function will be called again
        to adjust the fused layers.

        Returns
        -------
        None
        """

        self.latent_dim = self.mod2_dim

        self.enc_img_layer = nn.Linear(self.latent_dim, self.latent_dim)

        self.fused_dim = self.mod1_dim + self.latent_dim

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

        x_tab = x[0]
        x_encimg = x[1].squeeze(dim=1)
        # pass encimg into layer to turn it into half its size? or 64

        x_encimg = self.enc_img_layer(x_encimg)

        # concatenate with x_tab
        out_fuse = torch.cat((x_tab, x_encimg), dim=-1)

        # pass through fusion layers
        out_fuse = self.fused_layers(out_fuse)

        out = self.final_prediction(out_fuse)

        return [
            out,
        ]


class ImgLatentSpace(pl.LightningModule):
    """
    Pytorch lightning module: autoencoder to train the latent image space.

    Attributes
    ----------
    data_dims : dict
        Dictionary containing the dimensions of the data.
    img_dim : tuple
        Dimensions of the image data.
    latent_dim: int
        Dimension of the latent image space once we encode it down. Default is 64.
    encoder : nn.Sequential
        Sequential layer containing the encoder layers.
    decoder : nn.Sequential
        Sequential layer containing the decoder layers.
    new_encoder : nn.Sequential
        Sequential layer containing the encoder layers and the linear layer to reduce the dimension
        of the latent image space.
        Calculated with :meth:`~ImgLatentSpace.calc_fused_layers()`.
    new_decoder : nn.Sequential
        Sequential layer containing the decoder layers and the linear layer to increase the
        dimension of the latent image space.
        Calculated with :meth:`~ImgLatentSpace.calc_fused_layers()`.
    """

    def __init__(self, data_dims):
        """
        Parameters
        ----------
        data_dims : dict
            Dictionary containing the dimensions of the data.
        """
        super(ImgLatentSpace, self).__init__()

        self.data_dims = data_dims
        self.img_dim = data_dims[2]
        self.latent_dim = 64

        if len(self.img_dim) == 2:  # 2D images
            self.encoder = nn.Sequential(
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

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
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

        self.calc_fused_layers()

    def calc_fused_layers(self):
        """
        Calculate the fused layers. If layer sizes are modified, this function will be called again to adjust the
        fused layers.

        Returns
        -------
        None
        """

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

        # # release memory
        # self.encoder = None
        # self.decoder = None

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            Output of the model. Reconstruction/decoded image.
        """

        encoded = self.new_encoder(x)

        decoded = self.new_decoder(encoded)

        return decoded

    def training_step(self, batch, batch_idx):
        """
        Training step of the model.

        Parameters
        ----------
        batch : torch.Tensor
            Batch of data.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        torch.Tensor
            Loss of the model.
        """
        images = batch

        # Forward pass
        outputs = self(images)
        loss = nn.MSELoss()(outputs, images)

        self.log("train_loss", loss, logger=False)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step of the model.

        Parameters
        ----------
        batch : torch.Tensor
            Batch of data.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        torch.Tensor
            Loss of the model.
        """
        images = batch

        # Forward pass
        outputs = self(images)
        loss = nn.MSELoss()(outputs, images)

        self.log("val_loss", loss, logger=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizers of the model.

        Returns
        -------
        torch.optim.Adam
            Adam optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def encode_image(self, x):
        """
        Encode the image data. Used when the model is trained to get latent image space.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            Encoded image.
        """
        return self.new_encoder(x)


class concat_img_latent_tab_subspace_method:
    """
    Class containing the method to train the latent image space and to convert the image data
    to the latent image space.

    Attributes
    ----------
    datamodule : pl.LightningDataModule
        Data module containing the data.
    trainer : pytorch_lightning.Trainer
        Lightning trainer.
    autoencoder : ImgLatentSpace
        Autoencoder to train the latent image space.
    """

    def __init__(self, datamodule):
        """
        Parameters
        ----------
        datamodule : pl.LightningDataModule
            Data module containing the data.
        """
        self.datamodule = datamodule
        self.trainer = init_trainer(
            None, max_epochs=3
        )  # TODO change back to big number
        self.autoencoder = ImgLatentSpace(self.datamodule.data_dims)

    def train(self, train_dataset, val_dataset):
        """
        Train the latent image space.

        Parameters
        ----------
        train_dataset : Dataset
            Training dataset.
        val_dataset : Dataset
            Validation dataset.

        Returns
        -------
        list
            List containing the raw tabular data and the latent image space.
        pd.DataFrame
            Dataframe containing the labels.
        """
        tab_train = train_dataset[:][0]
        img_train = train_dataset[:][1]
        img_val = val_dataset[:][1]
        labels_train = train_dataset[:][2]

        train_dataloader = DataLoader(
            img_train.unsqueeze(dim=1), batch_size=8, shuffle=False
        )
        val_dataloader = DataLoader(
            img_val.unsqueeze(dim=1), batch_size=8, shuffle=False
        )

        self.trainer.fit(self.autoencoder, train_dataloader, val_dataloader)
        self.trainer.validate(self.autoencoder, val_dataloader)

        self.autoencoder.eval()

        encoded_imgs = self.autoencoder.encode_image(img_train.unsqueeze(1))

        return [tab_train, encoded_imgs.detach()], pd.DataFrame(
            labels_train, columns=["pred_label"]
        )

    def convert_to_latent(self, test_dataset):
        """
        Convert the image data to the latent image space.

        Parameters
        ----------
        test_dataset : Dataset
            Test dataset.

        Returns
        -------
        list
            List containing the raw tabular data and the latent image space.
        pd.DataFrame
            Dataframe containing the labels.
        list
            List containing the dimensions of the data.
        """
        tab_val = test_dataset[:][0]
        img_val = test_dataset[:][1]
        label_val = test_dataset[:][2]

        self.autoencoder.eval()

        encoded_imgs = self.autoencoder.encode_image(img_val.unsqueeze(1))

        return (
            [tab_val, encoded_imgs.detach()],
            pd.DataFrame(label_val, columns=["pred_label"]),
            [tab_val.shape[1], encoded_imgs.shape[1], None],
        )
