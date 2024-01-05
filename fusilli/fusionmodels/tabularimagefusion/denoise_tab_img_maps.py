"""
Denoising autoencoder for tabular data concatenated with image feature maps
"""
import torch.nn as nn
from fusilli.fusionmodels.base_model import ParentFusionModel
import torch
import lightning.pytorch as pl
from fusilli.utils.training_utils import (
    init_trainer,
    get_checkpoint_filenames_for_subspace_models,
)
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pandas as pd
from torch.nn import functional as F
from fusilli.fusionmodels.base_model import BaseModel

from fusilli.utils import check_model_validity


class DenoisingAutoencoder(pl.LightningModule):
    """
    Denoising autoencoder for tabular data: pytorch lightning module.

    Attributes
    ----------
    tab_dims : int
        Dimension of the input tabular data.
    upsampler : nn.Sequential
        Upsampling layers.
    downsampler : nn.Sequential
        Downsampling layers.
    loss : nn function
        Loss function. In this case, it's the mean squared error.
    """

    def __init__(self, data_dims):
        """
        Initialise the model.

        Parameters
        ----------
        data_dims : list
            List containing the dimensions of the data.

        """
        super().__init__()

        self.tab_dims = data_dims[0]
        self.latent_dim = 28 * 28

        self.upsampler = nn.Sequential(
            nn.Linear(self.tab_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.latent_dim),
            nn.ReLU(),
        )

        self.downsampler = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.tab_dims),
            nn.ReLU(),
        )

        self.calc_fused_layers()

        self.loss = nn.MSELoss()

    def calc_fused_layers(self):
        """
        Calculate the fused layers.

        Returns
        -------
        None
        """

        # this will change the upsampler and downsampler to be consistent with a modified latent dimension
        # you can also just change the upsampler and downsampler directly

        check_model_validity.check_dtype(self.upsampler, nn.Sequential, "upsampler")
        check_model_validity.check_dtype(self.downsampler, nn.Sequential, "downsampler")
        check_model_validity.check_dtype(self.latent_dim, int, "latent_dim")

        if self.latent_dim < 1:
            raise ValueError(
                "The latent dimension must be greater than 0. The latent dimension is currently: ",
                self.latent_dim,
            )

        self.upsampler[0] = nn.Linear(self.tab_dims, self.upsampler[0].out_features)
        self.upsampler[-2] = nn.Linear(
            self.upsampler[-2].in_features, self.latent_dim
        )  # -2 because of the relu

        self.downsampler[0] = nn.Linear(
            self.latent_dim, self.downsampler[0].out_features
        )
        self.downsampler[-2] = nn.Linear(  # -2 because of the relu
            self.downsampler[-2].in_features, self.tab_dims
        )

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        list
            List containing the output.
        """

        x_before_dropout = x

        # drop out 0.2 of the tabular data to 0
        # simulates missing data (adding noise)
        x_dropout = nn.Dropout(0.2)(x)

        # upsample
        x_latent = self.upsampler(x_dropout)

        # downsample
        out = self.downsampler(x_latent)

        # return reconstructed data and the non-dropped out data

        # return out, x_dropout

        return out, x_before_dropout

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Parameters
        ----------
        batch : torch.Tensor
            Input batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor
            Loss.
        """
        # loss is difference between input (non dropped out) and downsampled output

        x = batch

        output, x_dropout = self(x)

        loss = self.loss(output, x_dropout)

        self.log("train_loss", loss, logger=False)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Parameters
        ----------
        batch : torch.Tensor
            Input batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor
            Loss.
        """
        x = batch

        output, x_dropout = self(x)

        loss = self.loss(output, x_dropout)

        self.log("val_loss", loss, logger=False)

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

    def denoise(self, x):
        """
        Denoise the data to create the latent subspace.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            Latent subspace.
        """
        # don't do the dropout here, only in the training step

        # upsample
        x_latent = self.upsampler(x)
        x_latent.flatten()

        return x_latent.detach()


class ImgUnimodalDAE(pl.LightningModule):
    """
    Image unimodal network to go alongside the tabular denoising autoencoder: pytorch
    lightning module.

    Attributes
    ----------
    img_dim : int
        Dimension of the input image data.
    multiclass_dimensions : int
        Number of classes for multiclass classification.
    img_layers : nn.ModuleDict
        Image layers.
    num_layers : int
        Number of image layers.
    fused_dim : int
        Dimension of the fused layers.
        Final dimension of the image data after image layers.
    prediction_task : str
        Type of prediction.
    loss : function
        Loss function. Depends on the prediction type.
    fused_layers : nn.Sequential
        Fused layers.
    final_prediction : nn.Sequential
        Final prediction layers.
    loss : function
        Loss function. Depends on the prediction type.
    activation : function
        Activation function. Depends on the prediction type.

    """

    def __init__(self, data_dims, prediction_task, multiclass_dimensions):
        """
        Initialise the model.

        Parameters
        ----------
        data_dims : list
            List containing the dimensions of the data.
        prediction_task : str
            Type of prediction.
        multiclass_dimensions : int
            Number of classes for multiclass classification.
        """
        super().__init__()

        self.img_dim = data_dims[2]
        # needed for ParentFusionModel
        self.multiclass_dimensions = multiclass_dimensions
        self.prediction_task = prediction_task

        # get the img layers from ParentFusionModel
        ParentFusionModel.set_img_layers(self)  # this will set the img_layers

        if self.prediction_task == "regression":
            self.loss = lambda logits, y: nn.MSELoss()(logits, y.unsqueeze(dim=1))
            self.activation = lambda x: x
        elif self.prediction_task == "binary":
            self.loss = lambda logits, y: F.binary_cross_entropy_with_logits(
                logits, y.unsqueeze(dim=1).float()
            )
            self.activation = lambda x: torch.round(x).to(torch.int)

        elif self.prediction_task == "multiclass":
            self.loss = lambda logits, y: F.cross_entropy(
                BaseModel.safe_squeeze(logits),
                BaseModel.safe_squeeze(y).long(),
            )
            self.activation = lambda x: torch.argmax(nn.Softmax(dim=-1)(x), dim=-1)

        self.get_fused_dim()
        # self.fused_dim = list(self.img_layers.values())[-1][0].out_channels

        ParentFusionModel.set_fused_layers(self, fused_dim=self.fused_dim)

        # setting the final prediction layers
        self.calc_fused_layers()

    def get_fused_dim(self):
        """
        Get the dimension of the fused layers.

        """

        dummy_conv_output = Variable(torch.rand((1,) + tuple(self.img_dim)))
        for layer in self.img_layers.values():
            dummy_conv_output = layer(dummy_conv_output)
        self.fused_dim = dummy_conv_output.data.view(1, -1).size(1)

    def calc_fused_layers(self):
        """
        Calculate the fused layers.

        Returns
        -------
        None
        """

        check_model_validity.check_dtype(self.img_layers, nn.ModuleDict, "img_layers")
        check_model_validity.check_img_dim(self.img_layers, self.img_dim, "img_layers")

        self.num_layers = len(self.img_layers)

        self.get_fused_dim()

        # self.fused_dim = list(self.img_layers.values())[-1][0].out_channels

        # check fused layers
        self.fused_layers, out_dim = check_model_validity.check_fused_layers(
            self.fused_layers, self.fused_dim
        )

        ParentFusionModel.set_final_pred_layers(self, out_dim)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        list
            List containing the output.
        """
        # feed image data through conv network

        for i, layer in enumerate(self.img_layers.values()):
            x = layer(x)
        # flatten
        x = x.view(x.size(0), -1)

        # # linear layer to get it to 1280
        # x = self.linear(x)

        # feed through fused layers
        x = self.fused_layers(x)

        # feed through final pred layers
        out = self.final_prediction(x)

        return out

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Parameters
        ----------
        batch : torch.Tensor
            Input batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor
            Loss.
        """
        _, images, y = batch

        logits = self(images)

        loss = self.loss(
            logits.float().requires_grad_(True),
            y.float().requires_grad_(True),
        )

        self.log("train_loss", loss, logger=False)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Parameters
        ----------
        batch : torch.Tensor
            Input batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        torch.Tensor
            Loss.
        """
        _, images, y = batch

        logits = self(images)

        loss = self.loss(
            logits.float(),
            y.float(),
        )

        self.log("val_loss", loss, logger=False)

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

    def get_intermediate_featuremaps(self, x):
        """
        Get the intermediate feature maps to concatenate with the tabular latent subspace.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor
            Intermediate feature maps.
        """
        feature_maps = []

        for i, layer in enumerate(self.img_layers.values()):
            x = layer(x)

            if i >= self.num_layers - 2:
                # get the feature maps from the last 2 conv layers

                # view the feature maps as a 1D vector
                out_view = x.view(x.size(0), -1)

                feature_maps.append(out_view)

        concatenated_feature_maps = torch.cat(feature_maps, dim=1).detach()

        return concatenated_feature_maps


class denoising_autoencoder_subspace_method:
    """
    Class containing the method to train the denoising autoencoder and to convert the image data
    to the latent image space.

    Attributes
    ----------
    datamodule : pl.LightningDataModule
        Data module containing the data.
    dae_trainer : pl.Trainer
        Trainer for the denoising autoencoder.
    img_unimodal_trainer : pl.Trainer
        Trainer for the image unimodal network.
    autoencoder : DenoisingAutoencoder
        Tabular denoising autoencoder.
    img_unimodal : ImgUnimodalDAE
        Image unimodal network.
    """

    # adding the autoencoder and img_unimodal to the class so that we can access them later?

    subspace_models = [
        DenoisingAutoencoder,
        ImgUnimodalDAE,
    ]  # access later for loading checkpoint paths?

    def __init__(
            self,
            datamodule,
            k=None,
            max_epochs=1000,
            train_subspace=True,
    ):
        """
        Parameters
        ----------
        datamodule : pl.LightningDataModule
            Data module containing the data.
        k : int or None
            Number of subspaces. Default is None.
        max_epochs : int
            Maximum number of epochs. Default is 1000.
        train_subspace : bool
            Whether to train the subspace models. Default is True.
        """

        self.datamodule = datamodule

        checkpoint_filenames = get_checkpoint_filenames_for_subspace_models(self, k)

        self.autoencoder = self.subspace_models[0](self.datamodule.data_dims)

        self.img_unimodal = self.subspace_models[1](
            self.datamodule.data_dims,
            self.datamodule.prediction_task,
            self.datamodule.multiclass_dimensions,
        )

        # if train_subspace is True, then we are training the model.
        # else, we are loading the model for plotting with from_new_data
        if train_subspace:
            self.dae_trainer = init_trainer(
                logger=None,
                output_paths=self.datamodule.output_paths,
                max_epochs=max_epochs,
                checkpoint_filename=checkpoint_filenames[0],
                own_early_stopping_callback=self.datamodule.own_early_stopping_callback,

            )
            self.img_unimodal_trainer = init_trainer(
                logger=None,
                output_paths=self.datamodule.output_paths,
                max_epochs=max_epochs,
                checkpoint_filename=checkpoint_filenames[1],
                own_early_stopping_callback=self.datamodule.own_early_stopping_callback,
            )

    def load_ckpt(self, checkpoint_path):
        """
        Load the checkpoint of the subspace models

        Parameters
        ----------
        checkpoint_path : list
            Paths to the checkpoints. The checkpoint list must be a list of checkpoint elements containing the state
            dict of the subspace models.
        """
        # checkpoint1 = torch.load(checkpoint_path[0])
        # checkpoint2 = torch.load(checkpoint_path[1])

        self.autoencoder.load_state_dict(torch.load(checkpoint_path[0])["state_dict"])
        self.img_unimodal.load_state_dict(torch.load(checkpoint_path[1])["state_dict"])

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
        labels_train = train_dataset[:][2]

        # train and test DAE
        tab_train_dataloader = DataLoader(
            tab_train, batch_size=self.datamodule.batch_size, shuffle=False
        )
        tab_val_dataloader = DataLoader(
            tab_train, batch_size=self.datamodule.batch_size, shuffle=False
        )

        torch.set_grad_enabled(True)
        self.dae_trainer.fit(self.autoencoder, tab_train_dataloader, tab_val_dataloader)
        self.dae_trainer.validate(self.autoencoder, tab_val_dataloader)

        # -------train and test img unimodal----------------

        img_train_dataloader = DataLoader(
            train_dataset, batch_size=self.datamodule.batch_size, shuffle=False
        )
        img_val_dataloader = DataLoader(
            val_dataset, batch_size=self.datamodule.batch_size, shuffle=False
        )

        torch.set_grad_enabled(
            True
        )  # need to set this to true again after the DAE training
        self.img_unimodal_trainer.fit(
            self.img_unimodal, img_train_dataloader, img_val_dataloader
        )
        self.img_unimodal_trainer.validate(self.img_unimodal, img_val_dataloader)

        # ---------get latent outputs----------------

        self.autoencoder.eval()
        train_tab_latent_space = self.autoencoder.denoise(tab_train)

        self.img_unimodal.eval()
        train_img_feature_maps = self.img_unimodal.get_intermediate_featuremaps(
            img_train
        )

        # ---------concatenate them----------------
        train_latent_image_space = torch.cat(
            (train_tab_latent_space, train_img_feature_maps), dim=1
        )

        # save the trained trainers in dict
        self.trained_trainers = {
            self.autoencoder: self.dae_trainer,
            self.img_unimodal: self.img_unimodal_trainer,
        }

        # make the training dataset out of them
        return (
            train_latent_image_space,
            pd.DataFrame(
                labels_train,
                columns=["prediction_label"],
            ),
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

        # ---------DAE----------------
        self.autoencoder.eval()
        val_tab_latent_space = self.autoencoder.denoise(tab_val)

        # ---------img unimodal----------------
        self.img_unimodal.eval()
        val_img_feature_maps = self.img_unimodal.get_intermediate_featuremaps(img_val)

        # concatenate them
        val_latent_image_space = torch.cat(
            (val_tab_latent_space, val_img_feature_maps), dim=1
        )

        # make the training dataset out of them
        return (
            val_latent_image_space,
            pd.DataFrame(label_val, columns=["prediction_label"]),
            [val_latent_image_space.shape[1], None, None],
        )


class DAETabImgMaps(ParentFusionModel, nn.Module):
    """
    Using a denoising autoencoder to upsample tabular data, then concatenating with feature maps
    from final 3 conv layers of image data.
    From Yan et al 2021: Richer fusion network for breast cancer classification on multimodal data.

    Attributes
    ----------
    prediction_task : str
        Type of prediction.
    subspace_method : class
        Subspace method:
        :class:`~fusilli.fusion_models.denoise_tab_img_maps.denoising_autoencoder_subspace_method`.
    fusion_layers : nn.Sequential
        Fusion layers combining the intermediate image maps and the tabular latent subspace.
    final_prediction : nn.Sequential
        Final prediction layers.
    """

    #: str: Name of the method.
    method_name = "Denoising tabular autoencoder with image maps"
    #: str: Type of modality.
    modality_type = "tabular_image"
    #: str: Type of fusion.
    fusion_type = "subspace"
    # class: Subspace method.
    subspace_method = denoising_autoencoder_subspace_method

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

        self.fusion_layers = nn.Sequential(
            nn.Linear(self.mod1_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 64),
        )

        self.calc_fused_layers()

    def calc_fused_layers(self):
        """
        Calculate the fused layers.

        """

        check_model_validity.check_dtype(
            self.fusion_layers, nn.Sequential, "fusion_layers"
        )

        self.fusion_layers[0] = nn.Linear(
            self.mod1_dim, self.fusion_layers[0].out_features
        )

        self.set_final_pred_layers(self.fusion_layers[-1].out_features)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        list
            List containing the output.
        """

        check_model_validity.check_model_input(x, uni_modal_flag=True)

        x = self.fusion_layers(x)

        out = self.final_prediction(x)

        return [
            out,
        ]
