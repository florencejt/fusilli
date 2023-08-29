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


class ConcatImgLatentTabDoubleTrain(ParentFusionModel, nn.Module):
    """
    Concatenating the latent img space and tabular data. The latent image space is
    trained separately from both the tabular data and the labels, using the
    img_latent_subspace_method class.

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
    subspace_method : class
        Class containing the method to train the latent image space.
    new_encdim : int
        Dimension of the latent image space once we shrink it down.
    enc_img_layer : nn.Linear
        Linear layer to reduce the dimension of the latent image space.
    fused_dim : int
        Dimension of the fused layers.
    fused_layers : nn.Sequential
        Sequential layer containing the fused layers.
    final_prediction : nn.Sequential
        Sequential layer containing the final prediction layers. The final prediction layers
            take in the number of features of the fused layers as input.

    Methods
    -------
    forward(x)
        Forward pass of the model.

    """

    method_name = "Concatenating latent img space and tabular data training separately"
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
        self.subspace_method = img_latent_subspace_method

        new_encdim = 64
        self.enc_img_layer = nn.Linear(self.mod2_dim, new_encdim)

        self.fused_dim = self.mod1_dim + new_encdim
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
    img_dims : tuple
        Dimensions of the image data.
    encoder : nn.Sequential
        Sequential layer containing the encoder layers.
    decoder : nn.Sequential
        Sequential layer containing the decoder layers.

    Methods
    -------
    forward(x)
        Forward pass of the model.
    training_step(batch, batch_idx)
        Training step of the model.
    validation_step(batch, batch_idx)
        Validation step of the model.
    configure_optimizers()
        Configure the optimizers of the model.
    encode_image(x)
        Encode the image data. Used when the model is trained to get latent image space.
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
        self.img_dims = data_dims[2]

        if len(self.img_dims) == 2:  # 2D images
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                # nn.Conv3d(128, 256, kernel_size=3, stride=1),
                # nn.ReLU(),
            )

            self.decoder = nn.Sequential(
                # nn.ConvTranspose3d(256, 128, kernel_size=3, stride=1, output_padding=1),
                # nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1),
                nn.Sigmoid(),
                nn.Upsample(size=self.img_dims, mode="bilinear", align_corners=False),
            )
        elif len(self.img_dims) == 3:
            self.encoder = nn.Sequential(
                nn.Conv3d(1, 16, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Conv3d(16, 32, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Conv3d(32, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                # nn.Conv3d(128, 256, kernel_size=3, stride=1),
                # nn.ReLU(),
            )

            self.decoder = nn.Sequential(
                # nn.ConvTranspose3d(256, 128, kernel_size=3, stride=1, output_padding=1),
                # nn.ReLU(),
                nn.ConvTranspose3d(64, 32, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.ConvTranspose3d(32, 16, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.ConvTranspose3d(16, 1, kernel_size=3, stride=1),
                nn.Sigmoid(),
                nn.Upsample(size=self.img_dims, mode="trilinear", align_corners=False),
            )
        else:
            raise ValueError("Invalid image dimensions.")

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
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
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
        # images, _ = batch

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
        return self.encoder(x)


class img_latent_subspace_method:
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

    Methods
    -------
    train(train_dataset, val_dataset)
        Train the latent image space.
    convert_to_latent(test_dataset)
        Convert the image data to the latent image space.
    """

    def __init__(self, datamodule):
        """
        Parameters
        ----------
        datamodule : pl.LightningDataModule
            Data module containing the data.
        """
        self.datamodule = datamodule
        self.trainer = init_trainer(None, max_epochs=100)
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
        flatten_encoded_imgs = torch.flatten(encoded_imgs, start_dim=2)

        return [tab_train, flatten_encoded_imgs.detach()], pd.DataFrame(
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
        flatten_encoded_imgs = torch.flatten(encoded_imgs, start_dim=2).squeeze(dim=1)
        return (
            [tab_val, flatten_encoded_imgs.detach()],
            pd.DataFrame(label_val, columns=["pred_label"]),
            [tab_val.shape[1], flatten_encoded_imgs.shape[1], None],
        )
