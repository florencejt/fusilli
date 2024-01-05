"""
Concatenating the latent img space and tabular data. The latent image space is
trained separately from both the tabular data and the labels, using the
img_latent_subspace_method class.
"""

import copy

import pandas as pd
import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from fusilli.fusionmodels.base_model import ParentFusionModel
from fusilli.utils import check_model_validity
from fusilli.utils.training_utils import (
    get_checkpoint_filenames_for_subspace_models,
    init_trainer,
)
from fusilli.utils import model_modifier


class ImgLatentSpace(pl.LightningModule):
    """
    Pytorch lightning module: autoencoder to train the latent image space.

    Attributes
    ----------
    data_dims : dict
        List containing the dimensions of the data.
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
        data_dims : list
            List containing the dimensions of the data.
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

        check_model_validity.check_dtype(self.encoder, nn.Sequential, "encoder")
        check_model_validity.check_dtype(self.decoder, nn.Sequential, "decoder")
        check_model_validity.check_dtype(self.latent_dim, int, "latent dim")

        check_model_validity.check_img_dim(self.encoder, self.img_dim, "encoder")
        check_model_validity.check_img_dim(self.decoder, self.img_dim, "encoder")

        if self.latent_dim < 1:
            raise ValueError(
                f"Incorrect attribute range: The latent dimension must be greater than 0. The latent dimension is "
                f"currently: ",
                self.latent_dim,
            )

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

    subspace_models = [ImgLatentSpace]

    def __init__(self, datamodule, k=None, max_epochs=1000,
                 train_subspace=True):
        """
        Parameters
        ----------
        datamodule : class
            Data module containing the data.
        k : int or None
            Number of folds for cross validation. Default is None.
        max_epochs : int
            Maximum number of epochs to train the latent image space.
        train_subspace : bool
            Whether to train the latent image space or not.
            Default is True. If False, a new trainer will not be created. Then
            load_ckpt() must be called to load the checkpoint of the latent image space.
        """
        self.datamodule = datamodule

        self.autoencoder = ImgLatentSpace(self.datamodule.data_dims)

        if train_subspace:
            checkpoint_filenames = get_checkpoint_filenames_for_subspace_models(self, k)

            self.trainer = init_trainer(
                logger=None,
                output_paths=self.datamodule.output_paths,
                max_epochs=max_epochs,
                checkpoint_filename=checkpoint_filenames[0],
            )

    def load_ckpt(self, checkpoint_path):
        """
        Load the checkpoint of the latent image space.

        Parameters
        ----------
        checkpoint_path : list
            List containing the path to the checkpoint.

        """

        # load state dict only - of the already init-ed autoencoder
        self.autoencoder.load_state_dict(torch.load(checkpoint_path[0])["state_dict"])

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

        train_dataloader = DataLoader(img_train, batch_size=8, shuffle=False)
        # .unsqueeze(dim=1) removed

        val_dataloader = DataLoader(img_val, batch_size=8, shuffle=False)
        # .unsqueeze(dim=1) removed

        self.trainer.fit(self.autoencoder, train_dataloader, val_dataloader)
        self.trainer.validate(self.autoencoder, val_dataloader)

        self.autoencoder.eval()

        encoded_imgs = self.autoencoder.encode_image(img_train)
        # .unsqueeze(1)) removed

        return [tab_train, encoded_imgs.detach()], pd.DataFrame(
            labels_train, columns=["prediction_label"]
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

        encoded_imgs = self.autoencoder.encode_image(img_val)
        # .unsqueeze(1)) removed

        return (
            [tab_val, encoded_imgs.detach()],
            pd.DataFrame(label_val, columns=["prediction_label"]),
            [tab_val.shape[1], encoded_imgs.shape[1], None],
        )


class ConcatImgLatentTabDoubleTrain(ParentFusionModel, nn.Module):
    """
    Concatenating the latent img space and tabular data. The latent image space is
    trained separately from both the tabular data and the labels, using the
    img_latent_subspace_method class.

    Attributes
    ----------
    prediction_task : str
        Type of prediction to be performed.
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

    #: str: Name of the method.
    method_name = "Pretrained Latent Image + Tabular Data"
    #: str: Type of modality.
    modality_type = "tabular_image"
    #: str: Type of fusion.
    fusion_type = "subspace"
    #: class: Class containing the method to train the latent image space.
    subspace_method = concat_img_latent_tab_subspace_method

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

        self.fused_dim = self.mod1_dim + self.mod2_dim

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

        # check fused_layer
        self.get_fused_dim()
        self.fused_layers, out_dim = check_model_validity.check_fused_layers(
            self.fused_layers, self.fused_dim
        )

        self.set_final_pred_layers(out_dim)

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : tuple
            Tuple containing the input data. First element is the tabular data, second element is
            the image data from the image subspace method.

        Returns
        -------
        list
            List containing the output of the model.
        """

        # ~~ Checks ~~
        check_model_validity.check_model_input(x)

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
