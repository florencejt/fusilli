"""
This module implements the MCVAE (multi-channel variational autoencoder) model for fusing
two types of tabular data.
"""

import torch.nn as nn
from fusilli.fusion_models.base_model import ParentFusionModel
import torch
from fusilli.utils.mcvae.src.mcvae.models import Mcvae
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

    Attributes
    ----------
    datamodule : datamodule object
        Datamodule object containing the data.
    num_latent_dims : int
        Number of latent dimensions.
    fit_model : Mcvae object
        Mcvae object containing the fitted model.

    """

    subspace_models = [Mcvae]

    def __init__(self, datamodule, k=None, max_epochs=5000, checkpoint_path=None):
        """
        Parameters
        ----------
        datamodule : datamodule object
            Datamodule object containing the data.
        max_epochs : int, optional
            Maximum number of epochs, by default 5000
        checkpoint_path : list, optional
            List containing the path to the checkpoint, by default None
        """
        self.datamodule = datamodule
        self.num_latent_dims = 10
        self.max_epochs = max_epochs

        self.checkpoint_filenames = get_checkpoint_filenames_for_subspace_models(
            self, k=k
        )

        # load checkpoint if we're not training: plotting with new data
        print("checkpoint path", checkpoint_path)
        if checkpoint_path is not None:
            new_checkpoint_path = checkpoint_path[0][: -len(".ckpt")]
            checkpoint = torch.load(new_checkpoint_path)

            init_dict = {
                "n_channels": 2,
                "lat_dim": self.num_latent_dims,
                "n_feats": tuple([datamodule.data_dims[0], datamodule.data_dims[1]]),
            }

            self.fit_model = Mcvae(**init_dict, sparse=True)
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
                "Incorrect attribute range: The latent dimension must be greater than 0. The latent dimension is currently: ",
                self.num_latent_dims,
            )

    def get_latents(self, dataset):
        """
        Gets the latent representations of the multimodal dataset.
        The two latent spaces are averaged to form the joint latent space.

        Parameters
        ----------
        dataset : list
            List containing the two types of tabular data.

        Returns
        -------
        mean_latents : np.array
            Array containing the mean latents of the dataset.
        """
        # getting mean latent space

        q = self.fit_model.encode(dataset)

        latent_vars_ch0 = q[0].loc.detach().cpu()
        latent_vars_ch1 = q[1].loc.detach().cpu()
        latents = []

        n_dims = latent_vars_ch0.shape[1]

        for i in range(n_dims):
            latent_temp = np.vstack([latent_vars_ch0[:, i], latent_vars_ch1[:, i]])
            latents.append(np.mean(latent_temp, axis=0))

        indices = [i for i in range(self.num_latent_dims)]
        latents = [latents[i] for i in indices]

        mean_latents = np.vstack([latents]).transpose()  # 43 people

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
        labels = train_dataset[:][2]
        mcvae_training_data = [tab1, tab2]

        init_dict = {
            "n_channels": 2,
            "lat_dim": self.num_latent_dims,
            "n_feats": tuple(
                [mcvae_training_data[0].shape[1], mcvae_training_data[1].shape[1]]
            ),
        }
        mcvae_fit = Mcvae(**init_dict, sparse=True)
        mcvae_fit.init_loss()
        mcvae_fit.optimizer = torch.optim.Adam(mcvae_fit.parameters(), lr=0.001)

        with contextlib.redirect_stdout(None):
            mcvae_fit.optimize(epochs=self.max_epochs, data=mcvae_training_data)
            ideal_epoch = mcvae_early_stopping_tol(
                tolerance=3, patience=10, loss_logs=mcvae_fit.loss["total"]
            )

        mcvae_esfit = Mcvae(**init_dict, sparse=True)
        mcvae_esfit.init_loss()
        mcvae_esfit.optimizer = torch.optim.Adam(mcvae_esfit.parameters(), lr=0.001)
        with contextlib.redirect_stdout(None):
            mcvae_esfit.optimize(epochs=ideal_epoch, data=mcvae_training_data)

        self.fit_model = mcvae_esfit

        # save .ckpt file
        torch.save(
            self.fit_model.state_dict(),
            self.datamodule.params["checkpoint_dir"]
            + "/"
            + self.checkpoint_filenames[0],
        )

        # getting mean latent space
        mean_latents = self.get_latents(mcvae_training_data)

        return torch.Tensor(mean_latents), pd.DataFrame(labels, columns=["pred_label"])

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
        labels = test_dataset[:][2]
        mcvae_test_data = [tab1, tab2]

        test_mean_latents = self.get_latents(mcvae_test_data)

        return (
            torch.Tensor(test_mean_latents),
            pd.DataFrame(labels, columns=["pred_label"]),
            [self.num_latent_dims, None, None],
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

    # str: Name of the method.
    method_name = "MCVAE Tabular"
    # str: Type of modality.
    modality_type = "both_tab"
    # str: Type of fusion.
    fusion_type = "subspace"
    # class: Subspace method class.
    subspace_method = MCVAESubspaceMethod

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
        # self.subspace_method = MCVAESubspaceMethod

        self.latent_space_layers = nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(25, 32),
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

        # make sure the first layer takes in the latent dimension
        self.latent_space_layers["layer 1"][0] = nn.Linear(
            self.mod1_dim, self.latent_space_layers["layer 1"][0].out_features
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
        x : list
            List containing the two types of tabular data.

        Returns
        -------
        out_pred : list
            List containing the predictions.
        """
        x_latent = x

        for layer in self.latent_space_layers.values():
            x_latent = layer(x_latent)

        out_fuse = self.fused_layers(x_latent)

        out_pred = self.final_prediction(out_fuse)

        return [
            out_pred,
        ]