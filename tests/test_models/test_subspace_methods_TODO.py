import pytest
import torch
import pandas as pd
from unittest.mock import patch, Mock
import fusilli
# MCVAE subspace method
from fusilli.fusionmodels.tabularfusion.mcvae_model import MCVAESubspaceMethod


@pytest.fixture
def sample_datamodule():
    # Define a sample datamodule object for testing
    # You may need to create a fixture with appropriate data for your tests
    pass


@pytest.fixture
def sample_train_dataset():
    # Define a sample training dataset for testing
    pass


@pytest.fixture
def sample_test_dataset():
    # Define a sample test dataset for testing
    pass


# from fusilli.utils.training_utils import get_checkpoint_filenames_for_subspace_models


def test_mcvaesubspacemethod_initialisation(sample_datamodule, mocker):
    # mock the fusilli.utils.training_utils.get_checkpoint_filenames_for_subspace_models
    # function to return a list of checkpoint filenames

    mocker.patch(
        "fusilli.utils.training_utils.get_checkpoint_filenames_for_subspace_models",
        return_value=["model1.pt", "model2.pt"]
    )
    # patcher = patch(
    #     "fusilli.utils.training_utils.get_checkpoint_filenames_for_subspace_models",
    #     return_value=["model1.pt", "model2.pt"],
    # )

    # before initialisation
    assert hasattr(MCVAESubspaceMethod, "subspace_models")

    with patch("fusilli.utils.training_utils.get_checkpoint_filenames_for_subspace_models") as mock:
        mock.return_value = ["model1.pt", "model2.pt"]
        mcvae_subspace = MCVAESubspaceMethod(sample_datamodule)

        assert mcvae_subspace.datamodule == sample_datamodule
        assert mcvae_subspace.num_latent_dims == 10


def test_mcvaesubspacemethod_check_params(sample_datamodule):
    # Test the check_params method
    mcvae_subspace = MCVAESubspaceMethod(sample_datamodule)
    mcvae_subspace.check_params()  # Ensure it doesn't raise exceptions


def test_mcvaesubspacemethod_train(sample_train_dataset):
    # Test the train method
    mcvae_subspace = MCVAESubspaceMethod(sample_datamodule)
    mean_latents, labels = mcvae_subspace.train(sample_train_dataset)
    assert isinstance(mean_latents, torch.Tensor)
    assert isinstance(labels, pd.DataFrame)
    # Add more assertions as needed


def test_mcvaesubspacemethod_convert_to_latent(sample_test_dataset):
    # Test the convert_to_latent method
    mcvae_subspace = MCVAESubspaceMethod(sample_datamodule)
    test_mean_latents, labels, dimensions = mcvae_subspace.convert_to_latent(sample_test_dataset)
    assert isinstance(test_mean_latents, torch.Tensor)
    assert isinstance(labels, pd.DataFrame)
    assert isinstance(dimensions, list)
