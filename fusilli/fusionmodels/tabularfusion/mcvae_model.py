"""
This module implements the MCVAE (multi-channel variational autoencoder) model for fusing
two types of tabular data.
"""

import torch.nn as nn
from fusilli.fusionmodels.base_model import ParentFusionModel
import torch

# from fusilli.utils.mcvae.src.mcvae.models import Mcvae

# FOR LOCAL INSTALL - CHANGE WHEN PUSHING TO RELEASE
from mcvae.models import Mcvae

import contextlib
import pandas as pd
import numpy as np
from fusilli.utils.training_utils import get_checkpoint_filenames_for_subspace_models
import sys

from fusilli.utils import check_model_validity


def mcvae_early_stopping_tol(patience, tolerance, loss_logs, verbose=False):
    """
    Simple early stopping function for the MCVAE model's training.

    Parameters
    ----------
    patience: int
        Number of epochs to wait before stopping
    tolerance: int
        Tolerance for loss
    loss_logs: list
        List of loss logs
    verbose: bool
        Whether to print out information

    Returns
    -------
    i: int
        Epoch to stop at
    """

    last_loss = -2000
    triggertimes = 0
    done = 0
    i = 0
    for i in range(len(loss_logs)):
        current_loss = loss_logs[i]

        if abs(current_loss - last_loss) < tolerance:
            triggertimes += 1

            if triggertimes >= patience:
                if verbose:
                    print(
                        f"Epoch chosen after early stopping with patience {patience} \
                    and tolerance {tolerance} : {i - triggertimes}.",
                        file=sys.stdout,
                    )
                done = 1
                break

        else:
            triggertimes = 0

        last_loss = current_loss

    if done == 0:
        if verbose:
            print("No epoch chosen with this patience.")

    return i - triggertimes


class MCVAESubspaceMethod:
    """
    Class for creating the MCVAE (multi-channel variational autoencoder) joint latent space.

    If you want to change the tolerance or patience for early stopping, you can do so by adding
    extra keyword arguments to the function prepare_fusion_data. For example:

    "mcvae_patience": value,
    "mcvae_tolerance": value,

    where value is the number of epochs for patience and the tolerance for tolerance.


    Attributes
    ----------
    datamodule : datamodule object
        Datamodule object containing the data.
    num_latent_dims : int
        Number of latent dimensions.
    fit_model : Mcvae object
        Mcvae object containing the fitted model.
    device : torch.device
        Device to run the model on.
    max_epochs : int
        Maximum number of epochs.
    checkpoint_filenames : list
        List containing the checkpoint filenames.
    mcvae_patience : int
        Number of epochs to wait before stopping training. By default, 10 but can be changed through the layer_mods dictionary input into ``prepare_fusion_data``.
    mcvae_tolerance : int
        Tolerance for loss. By default, 3 but can be changed through the layer_mods dictionary input into ``prepare_fusion_data``.


    """

    subspace_models = [Mcvae]

    def __init__(self, datamodule, k=None, max_epochs=5000, train_subspace=True):
        """
        Parameters
        ----------
        datamodule : datamodule object
            Datamodule object containing the data.
        k : int, optional
            Number of latent dimensions, by default None
        max_epochs : int, optional
            Maximum number of epochs, by default 5000
        train_subspace : bool, optional
            Whether to train the subspace model, by default True.
        """
        self.datamodule = datamodule
        self.device = torch.device("cpu")

        # default latent dim
        self.num_latent_dims = 10
        self.mcvae_patience = 10
        self.mcvae_tolerance = 3
        if self.datamodule.layer_mods is not None:
            # if MCVAESubspaceMethod is in the keys
            if "MCVAESubspaceMethod" in self.datamodule.layer_mods.keys():
                if (
                    "num_latent_dims"
                    in self.datamodule.layer_mods["MCVAESubspaceMethod"]
                ):
                    self.num_latent_dims = self.datamodule.layer_mods[
                        "MCVAESubspaceMethod"
                    ]["num_latent_dims"]

                # early stopping
                # PATIENCE
                if "patience" in self.datamodule.layer_mods["MCVAESubspaceMethod"]:
                    self.mcvae_patience = self.datamodule.layer_mods[
                        "MCVAESubspaceMethod"
                    ]["patience"]

                # TOLERANCE
                if "tolerance" in self.datamodule.layer_mods["MCVAESubspaceMethod"]:
                    self.mcvae_tolerance = self.datamodule.layer_mods[
                        "MCVAESubspaceMethod"
                    ]["tolerance"]

        self.max_epochs = max_epochs

        self.checkpoint_filenames = get_checkpoint_filenames_for_subspace_models(
            self, k=k
        )

    def load_ckpt(self, checkpoint_path):
        """
        Loads the checkpoint.

        Parameters
        ----------
        checkpoint_path : list
            List containing the path to the checkpoint.

        Returns
        -------
        None
        """
        new_checkpoint_path = checkpoint_path[0][: -len(".ckpt")]
        checkpoint = torch.load(new_checkpoint_path)

        if self.datamodule.data_dims["mod3_dim"] is not None:
            n_channels = 3
            n_features = tuple(
                [
                    self.datamodule.data_dims["mod1_dim"],
                    self.datamodule.data_dims["mod2_dim"],
                    self.datamodule.data_dims["mod3_dim"],
                ]
            )
        else:
            n_channels = 2
            n_features = tuple(
                [
                    self.datamodule.data_dims["mod1_dim"],
                    self.datamodule.data_dims["mod2_dim"],
                ]
            )

        init_dict = {
            "n_channels": n_channels,
            "lat_dim": self.num_latent_dims,
            "n_feats": n_features,
        }

        self.fit_model = Mcvae(**init_dict, sparse=True)
        self.fit_model.to(self.device)
        self.fit_model.load_state_dict(checkpoint)

    def check_params(self):
        """
        Checks the parameters of the model.

        Returns
        -------
        None
        """

        check_model_validity.check_dtype(self.num_latent_dims, int, "num_latent_dims")

        if self.num_latent_dims < 0:
            raise ValueError(
                "Incorrect attribute range: The latent dimension must be greater than 0. "
                "The latent dimension is currently: ",
                self.num_latent_dims,
            )

    def get_latents(self, dataset):
        """
        Gets the latent representations of the multimodal dataset.
        The two latent spaces are averaged to form the joint latent space.

        Parameters
        ----------
        dataset : list
            List containing the types of tabular data.

        Returns
        -------
        mean_latents : np.array
            Array containing the mean latents of the dataset.
        """

        # how many channels are in this method (2 or 3, equal to number of tabular modalities)
        num_channels = len(dataset)

        q = self.fit_model.encode(dataset)

        latent_vars_ch0 = q[0].loc.detach().cpu()
        latent_vars_ch1 = q[1].loc.detach().cpu()
        if num_channels == 3:
            latent_vars_ch2 = q[2].loc.detach().cpu()
        latents = []

        n_dims = latent_vars_ch0.shape[1]

        for i in range(n_dims):
            if num_channels == 3:
                latent_temp = np.vstack(
                    [
                        latent_vars_ch0[:, i],
                        latent_vars_ch1[:, i],
                        latent_vars_ch2[:, i],
                    ]
                )
            else:
                latent_temp = np.vstack([latent_vars_ch0[:, i], latent_vars_ch1[:, i]])
            latents.append(np.mean(latent_temp, axis=0))

        indices = [i for i in range(self.num_latent_dims)]
        latents = [latents[i] for i in indices]

        mean_latents = np.vstack([latents]).transpose()

        return mean_latents

    def train(self, train_dataset, val_dataset=None):
        """
        Trains the model.

        Parameters
        ----------
        train_dataset : list
            List containing the two types of tabular data.
        val_dataset : list, optional
            List containing the two types of tabular data, by default None

        Returns
        -------
        mean_latents : torch.Tensor
            Tensor containing the mean latents of the dataset.
        labels : pd.DataFrame
            Dataframe containing the labels of the dataset.
        """
        tab1 = train_dataset[:][0]
        tab2 = train_dataset[:][1]
        if len(train_dataset[:]) == 4:
            tab3 = train_dataset[:][2]
        labels = train_dataset[:][-1]

        # turn tab1 into a tensor
        tab1 = torch.Tensor(tab1).to(self.device)
        tab2 = torch.Tensor(tab2).to(self.device)
        mcvae_training_data = [tab1.to(self.device), tab2.to(self.device)]

        if len(train_dataset[:]) == 4:
            tab3 = torch.Tensor(tab3).to(self.device)
            mcvae_training_data.append(tab3.to(self.device))

        # send tab1 to device and everything in tab1
        # for i in range(len(tab1)):
        #     tab1[i] = tab1[i].to(self.device)
        #     tab2[i] = tab2[i].to(self.device)

        # mcvae_training_data = [tab1.to(self.device), tab2.to(self.device)]

        num_channels = len(mcvae_training_data)
        num_features = [mcvae_training_data[i].shape[1] for i in range(num_channels)]

        init_dict = {
            "n_channels": num_channels,
            "lat_dim": self.num_latent_dims,
            "n_feats": num_features,
        }
        mcvae_fit = Mcvae(**init_dict, sparse=True)

        mcvae_fit.init_loss()

        mcvae_fit.optimizer = torch.optim.Adam(mcvae_fit.parameters(), lr=0.001)
        mcvae_fit.to(self.device)

        with contextlib.redirect_stdout(None):
            mcvae_fit.optimize(epochs=self.max_epochs, data=mcvae_training_data)
            ideal_epoch = mcvae_early_stopping_tol(
                tolerance=self.mcvae_tolerance,
                patience=self.mcvae_patience,
                loss_logs=mcvae_fit.loss["total"],
            )

        mcvae_esfit = Mcvae(**init_dict, sparse=True)
        mcvae_esfit.to(self.device)
        mcvae_esfit.init_loss()
        mcvae_esfit.optimizer = torch.optim.Adam(mcvae_esfit.parameters(), lr=0.001)
        with contextlib.redirect_stdout(None):
            mcvae_esfit.optimize(epochs=ideal_epoch, data=mcvae_training_data)

        self.fit_model = mcvae_esfit

        # save .ckpt file
        torch.save(
            self.fit_model.state_dict(),
            self.datamodule.output_paths["checkpoints"]
            + "/"
            + self.checkpoint_filenames[0],
        )

        # getting mean latent space
        mean_latents = self.get_latents(mcvae_training_data)

        return torch.Tensor(mean_latents), pd.DataFrame(
            labels, columns=["prediction_label"]
        )

    def convert_to_latent(self, test_dataset):
        """
        Converts the test dataset to the latent space.

        Parameters
        ----------
        test_dataset : list
            List containing the two types of tabular data.

        Returns
        -------
        test_mean_latents : torch.Tensor
            Tensor containing the mean latents of the dataset.
        labels : pd.DataFrame
            Dataframe containing the labels of the dataset.
        [self.num_latent_dims, None, None] : list
            List containing the dimensions of the data.
        """
        tab1 = test_dataset[:][0]
        tab2 = test_dataset[:][1]
        mcvae_test_data = [tab1.to(self.device), tab2.to(self.device)]
        if len(test_dataset[:]) == 4:
            tab3 = test_dataset[:][2]
            mcvae_test_data.append(tab3.to(self.device))
        labels = test_dataset[:][-1]

        test_mean_latents = self.get_latents(mcvae_test_data)

        return (
            torch.Tensor(test_mean_latents),
            pd.DataFrame(labels, columns=["prediction_label"]),
            {
                "mod1_dim": self.num_latent_dims,
                "mod2_dim": None,
                "mod3_dim": None,
                "img_dim": None,
            },
        )


class MCVAE_tab(ParentFusionModel, nn.Module):
    """
    This class implements a model that fuses the two types of tabular data using the MCVAE approach.
    MCVAE: multi-channel variational autoencoder.

    The MCVAE creates a joint latent space of the two types of tabular data based off a joint
    latent prior and joint decoding.


    References
    ----------

    Antelmi, L., Ayache, N., Robert, P., & Lorenzi, M. (2019). Sparse Multi-Channel Variational
        Autoencoder for the Joint Analysis of Heterogeneous Data. Proceedings of the 36th
        International Conference on Machine Learning, 302–311.
        https://proceedings.mlr.press/v97/antelmi19a.html

    Attributes
    ----------
    subspace_method : class
        Class of the subspace method: :class:`~.MCVAESubspaceMethod`
    latent_space_layers : dict
        Dictionary containing the layers of the 1st type of tabular data.
        Here the first type of tabular data is the joint latent space created
        in the mcvae_subspace_method class.
    fused_dim : int
        Number of features of the fused layers. This is the flattened output size of the
        latent space layers.
    fused_layers : nn.Sequential
        Sequential layer containing the fused layers.
    final_prediction : nn.Sequential
        Sequential layer containing the final prediction layers.

    """

    #: str: Name of the method.
    method_name = "MCVAE Tabular"
    #: str: Type of modality.
    modality_type = "tabular_tabular"
    #: str: Type of fusion.
    fusion_type = "subspace"
    # class: Subspace method class.
    subspace_method = MCVAESubspaceMethod
    #: str: Available for three tabular modalities.
    three_modalities = True

    def __init__(self, prediction_task, data_dims, multiclass_dimensions):
        """
        Parameters
        ----------
        prediction_task : str
            Type of prediction to be performed.
        data_dims : dict
            Dictionary of data dimensions with keys "mod1_dim", "mod2_dim", "mod3_dim", and "img_dim".
        multiclass_dimensions : int
            Number of classes in the multiclass classification task.
        """
        ParentFusionModel.__init__(
            self, prediction_task, data_dims, multiclass_dimensions
        )

        self.prediction_task = prediction_task

        self.latent_space_layers = nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(data_dims["mod1_dim"], 32),
                    nn.ReLU(),
                ),
                "layer 2": nn.Sequential(
                    nn.Linear(32, 64),
                    nn.ReLU(),
                ),
                "layer 3": nn.Sequential(
                    nn.Linear(64, 128),
                    nn.ReLU(),
                ),
                "layer 4": nn.Sequential(
                    nn.Linear(128, 256),
                    nn.ReLU(),
                ),
                "layer 5": nn.Sequential(
                    nn.Linear(256, 256),
                    nn.ReLU(),
                ),
            }
        )
        self.fused_dim = list(self.latent_space_layers.values())[-1][0].out_features
        self.set_fused_layers(self.fused_dim)
        self.calc_fused_layers()

    def calc_fused_layers(self):
        """
        Calculates the fused layers of the model.

        Returns
        -------
        None
        """

        check_model_validity.check_dtype(
            self.latent_space_layers, nn.ModuleDict, "latent_space_layers"
        )

        # check fused layers
        self.fused_dim = list(self.latent_space_layers.values())[-1][0].out_features
        self.fused_layers, out_dim = check_model_validity.check_fused_layers(
            self.fused_layers, self.fused_dim
        )

        self.set_final_pred_layers(out_dim)

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            torch.Tensor containing the input data: joint latent space of the two types of tabular data.

        Returns
        -------
        torch.Tensor
            Tensor containing the predicted values.
        """
        # ~~ Checks ~~
        check_model_validity.check_model_input(x)

        x_latent = x

        for layer in self.latent_space_layers.values():
            x_latent = layer(x_latent)

        out_fuse = self.fused_layers(x_latent)

        out_pred = self.final_prediction(out_fuse)

        return out_pred
