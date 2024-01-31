import pytest
import torch.nn as nn
from fusilli.utils import model_modifier
from fusilli.fusionmodels.tabularimagefusion.denoise_tab_img_maps import (
    denoising_autoencoder_subspace_method,
    DAETabImgMaps,
)
from fusilli.fusionmodels.tabularimagefusion.concat_img_latent_tab_doubleloss import (
    ConcatImgLatentTabDoubleLoss,
)
from fusilli.fusionmodels.tabularimagefusion.concat_img_latent_tab_doubletrain import (
    concat_img_latent_tab_subspace_method,
    ConcatImgLatentTabDoubleTrain,
)
# from fusilli.fusionmodels.tabularfusion.mcvae_model import (
#     MCVAESubspaceMethod,
#     MCVAE_tab,
# )

from tests.test_data.test_TrainTestDataModule import create_test_files
from fusilli.data import TrainTestDataModule
import warnings
from unittest.mock import patch, Mock

# correct modifications
correct_modifications_2D = {
    "all": {
        "mod1_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(20, 32),
                    nn.ReLU(),
                ),
                "layer 2": nn.Sequential(
                    nn.Linear(32, 66),
                    nn.ReLU(),
                ),
                "layer 3": nn.Sequential(
                    nn.Linear(66, 128),
                    nn.ReLU(),
                ),
            }
        ),
        "img_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Conv2d(1, 35, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
                "layer 2": nn.Sequential(
                    nn.Conv2d(35, 70, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
                "layer 3": nn.Sequential(
                    nn.Conv2d(70, 128, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
            }
        ),
    },
    "MCVAESubspaceMethod": {
        "num_latent_dims": 20,
    },
    "ConcatImgLatentTabDoubleLoss": {
        "latent_dim": 50,
        "fused_layers": nn.Sequential(
            nn.Linear(25, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
    },
    "ConcatImgLatentTabDoubleTrain": {
        "fused_layers": nn.Sequential(
            nn.Linear(25, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
    },
    "MCVAE_tab": {
        "latent_space_layers": nn.ModuleDict(
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
                    nn.Linear(128, 310),
                    nn.ReLU(),
                ),
            }
        ),
        "fused_layers": nn.Sequential(
            nn.Linear(25, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
    },
    "DAETabImgMaps": {
        "fusion_layers": nn.Sequential(
            nn.Linear(20, 420),
            nn.ReLU(),
            nn.Linear(420, 100),
            nn.ReLU(),
            nn.Linear(100, 78),
        ),
    },
    "denoising_autoencoder_subspace_method": {
        "autoencoder.latent_dim": 180,  # denoising autoencoder latent dim
        "autoencoder.upsampler": nn.Sequential(
            nn.Linear(25, 80),
            nn.ReLU(),
            nn.Linear(80, 100),
            nn.ReLU(),
            nn.Linear(100, 150),
            nn.ReLU(),
        ),
        "autoencoder.downsampler": nn.Sequential(
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, 80),
            nn.ReLU(),
            nn.Linear(80, 20),
            nn.ReLU(),
        ),
        "img_unimodal.img_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Conv2d(1, 40, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
                "layer 2": nn.Sequential(
                    nn.Conv2d(40, 60, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
                "layer 3": nn.Sequential(
                    nn.Conv2d(60, 85, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
            }
        ),
        "img_unimodal.fused_layers": nn.Sequential(
            nn.Linear(25, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
    },
    "concat_img_latent_tab_subspace_method": {
        "autoencoder.latent_dim": 180,  # img unimodal autoencoder
        "autoencoder.encoder": nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ),
        "autoencoder.decoder": nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
        ),
    },
}

correct_modifications_3D = {
    "all": {
        "mod1_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(20, 32),
                    nn.ReLU(),
                ),
                "layer 2": nn.Sequential(
                    nn.Linear(32, 66),
                    nn.ReLU(),
                ),
                "layer 3": nn.Sequential(
                    nn.Linear(66, 128),
                    nn.ReLU(),
                ),
            }
        ),
        "img_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Conv3d(1, 10, kernel_size=(3, 3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool3d((2, 2, 2)),
                ),
                "layer 2": nn.Sequential(
                    nn.Conv3d(10, 25, kernel_size=(3, 3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool3d((2, 2, 2)),
                ),
                "layer 3": nn.Sequential(
                    nn.Conv3d(25, 40, kernel_size=(3, 3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool3d((2, 2, 2)),
                ),
            }
        ),
    },
    "MCVAESubspaceMethod": {
        "num_latent_dims": 20,
    },
    "ConcatImgLatentTabDoubleLoss": {
        "latent_dim": 50,
        "fused_layers": nn.Sequential(
            nn.Linear(25, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
    },
    "MCVAE_tab": {
        "latent_space_layers": nn.ModuleDict(
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
                    nn.Linear(128, 310),
                    nn.ReLU(),
                ),
            }
        ),
        "fused_layers": nn.Sequential(
            nn.Linear(25, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
    },
    "DAETabImgMaps": {
        "fusion_layers": nn.Sequential(
            nn.Linear(20, 420),
            nn.ReLU(),
            nn.Linear(420, 100),
            nn.ReLU(),
            nn.Linear(100, 78),
        ),
    },
    "denoising_autoencoder_subspace_method": {
        "autoencoder.latent_dim": 180,  # denoising autoencoder latent dim
        "autoencoder.upsampler": nn.Sequential(
            nn.Linear(25, 80),
            nn.ReLU(),
            nn.Linear(80, 100),
            nn.ReLU(),
            nn.Linear(100, 150),
            nn.ReLU(),
        ),
        "autoencoder.downsampler": nn.Sequential(
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, 80),
            nn.ReLU(),
            nn.Linear(80, 20),
            nn.ReLU(),
        ),
        "img_unimodal.img_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Conv3d(1, 10, kernel_size=(3, 3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool3d((2, 2, 2)),
                ),
                "layer 2": nn.Sequential(
                    nn.Conv3d(10, 25, kernel_size=(3, 3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool3d((2, 2, 2)),
                ),
                "layer 3": nn.Sequential(
                    nn.Conv3d(25, 40, kernel_size=(3, 3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool3d((2, 2, 2)),
                ),
            }
        ),
        "img_unimodal.fused_layers": nn.Sequential(
            nn.Linear(25, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
    },
    "concat_img_latent_tab_subspace_method": {
        "autoencoder.latent_dim": 180,  # img unimodal autoencoder
        "autoencoder.encoder": nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(8, 15, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(15, 35, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        ),
        "autoencoder.decoder": nn.Sequential(
            # nn.ConvTranspose3d(256, 128, kernel_size=3, stride=1, output_padding=1),
            # nn.ReLU(),
            nn.ConvTranspose3d(35, 15, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose3d(15, 8, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose3d(8, 1, kernel_size=3, stride=1),
        ),
    },
    "ConcatImgLatentTabDoubleTrain": {
        "fused_layers": nn.Sequential(
            nn.Linear(25, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
    },
}

# wrong_layer_type_modifications
wrong_layer_type_modifications = {
    "all": {
        "mod1_layers": nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 66),
            nn.ReLU(),
            nn.Linear(66, 128),
            nn.ReLU(),
        ),
        "img_layers": nn.Sequential(
            nn.Conv2d(1, 35, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(35, 70, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(70, 128, kernel_size=(3, 3), padding=0),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        ),
    },
    "ConcatImgLatentTabDoubleLoss": {
        "latent_dim": 50.5,
        "fused_layers": nn.ModuleDict(
            {
                "layer1": nn.Linear(25, 150),
                "relu1": nn.ReLU(),
                "layer2": nn.Linear(150, 75),
                "relu2": nn.ReLU(),
                "layer3": nn.Linear(75, 50),
                "relu3": nn.ReLU(),
            }
        ),
    },
    "MCVAE_tab": {
        "latent_space_layers": nn.Sequential(
            nn.Linear(25, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 310),
            nn.ReLU(),
        ),
        "fused_layers": nn.ModuleDict(
            {
                "layer1": nn.Linear(25, 150),
                "relu1": nn.ReLU(),
                "layer2": nn.Linear(150, 75),
                "relu2": nn.ReLU(),
                "layer3": nn.Linear(75, 50),
                "relu3": nn.ReLU(),
            }
        ),
    },
    "DAETabImgMaps": {
        "fusion_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(nn.Linear(20, 420), nn.ReLU()),
                "layer 2": nn.Sequential(nn.Linear(420, 100), nn.ReLU()),
                "layer 3": nn.Sequential(nn.Linear(100, 78)),
            },
        )
    },
    "MCVAESubspaceMethod": {
        "num_latent_dims": 20.5,
    },
    "denoising_autoencoder_subspace_method": {
        "autoencoder.latent_dim": 180.75,  # denoising autoencoder latent dim
        "autoencoder.upsampler": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(nn.Linear(25, 80), nn.ReLU()),
                "layer 2": nn.Sequential(nn.Linear(80, 100), nn.ReLU()),
                "layer 3": nn.Sequential(nn.Linear(100, 150), nn.ReLU()),
            }
        ),
        "autoencoder.downsampler": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(nn.Linear(150, 100), nn.ReLU()),
                "layer 2": nn.Sequential(nn.Linear(100, 80), nn.ReLU()),
                "layer 3": nn.Sequential(nn.Linear(80, 20), nn.ReLU()),
            }
        ),
        "img_unimodal.img_layers": nn.Sequential(
            nn.Conv3d(1, 10, kernel_size=(3, 3, 3), padding=0),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(10, 25, kernel_size=(3, 3, 3), padding=0),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(25, 40, kernel_size=(3, 3, 3), padding=0),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
        ),
        "img_unimodal.fused_layers": nn.ModuleDict(
            {
                "layer1": nn.Linear(25, 150),
                "relu1": nn.ReLU(),
                "layer2": nn.Linear(150, 75),
                "relu2": nn.ReLU(),
                "layer3": nn.Linear(75, 50),
                "relu3": nn.ReLU(),
            }
        ),
    },
    "concat_img_latent_tab_subspace_method": {
        "autoencoder.latent_dim": 180.25,  # img unimodal autoencoder
        "autoencoder.encoder": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Conv3d(1, 8, kernel_size=3, stride=1),
                    nn.ReLU(),
                    nn.MaxPool3d(kernel_size=2, stride=2),
                ),
                "layer 2": nn.Sequential(
                    nn.Conv3d(8, 15, kernel_size=3, stride=1),
                    nn.ReLU(),
                    nn.MaxPool3d(kernel_size=2, stride=2),
                ),
                "layer 3": nn.Sequential(
                    nn.Conv3d(15, 35, kernel_size=3, stride=1),
                    nn.ReLU(),
                    nn.MaxPool3d(kernel_size=2, stride=2),
                ),
            }
        ),
        "autoencoder.decoder": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.ConvTranspose3d(35, 15, kernel_size=3, stride=1),
                    nn.ReLU(),
                ),
                "layer 2": nn.Sequential(
                    nn.ConvTranspose3d(15, 8, kernel_size=3, stride=1),
                    nn.ReLU(),
                ),
                "layer 3": nn.Sequential(
                    nn.ConvTranspose3d(8, 1, kernel_size=3, stride=1),
                ),
            }
        ),
    },
    "ConcatImgLatentTabDoubleTrain": {
        "fused_layers": nn.ModuleDict(
            {
                "layer1": nn.Linear(25, 150),
                "relu1": nn.ReLU(),
                "layer2": nn.Linear(150, 75),
                "relu2": nn.ReLU(),
                "layer3": nn.Linear(75, 50),
                "relu3": nn.ReLU(),
            }
        ),
    },
}


# ~~~~~~~~~~ FIXTURES ~~~~~~~~~~ #
@pytest.fixture
def model_instance_DAETabImgMaps():
    prediction_task = "regression"
    data_dims = [10, 15, [100, 100]]
    multiclass_dimensions = None
    return DAETabImgMaps(prediction_task, data_dims, multiclass_dimensions)


@pytest.fixture
def model_instance_ConcatImgLatentTabDoubleLoss():
    prediction_task = "regression"
    data_dims = [10, None, [100, 100]]
    multiclass_dimensions = None
    return ConcatImgLatentTabDoubleLoss(prediction_task, data_dims, multiclass_dimensions)


@pytest.fixture
def model_instance_ConcatImgLatentTabDoubleTrain():
    prediction_task = "regression"
    data_dims = [10, 15, None]  # represents mod1 dim and img latent space dim
    multiclass_dimensions = None
    return ConcatImgLatentTabDoubleTrain(prediction_task, data_dims, multiclass_dimensions)


# @pytest.fixture
# def model_instance_MCVAE_tab():
#     prediction_task = "regression"
#     data_dims = [10, 15, None]
#     multiclass_dimensions = None
#     return MCVAE_tab(prediction_task, data_dims, params)


model_instances_training = [
    ("DAETabImgMaps", "model_instance_DAETabImgMaps"),
    ("ConcatImgLatentTabDoubleLoss", "model_instance_ConcatImgLatentTabDoubleLoss"),
    # ("MCVAE_tab", "model_instance_MCVAE_tab"),
    ("ConcatImgLatentTabDoubleTrain", "model_instance_ConcatImgLatentTabDoubleTrain"),
]


# testing correct modifications: tabular-only or 2D
@pytest.mark.parametrize("model_name, model_fixture", model_instances_training)
def test_correct_modify_model_architecture_2D_training(
        model_name, model_fixture, request
):
    # "modification" may have been modified (ironically) by the calc_fused_layers method
    # in some of the models. This is to ensure that the input to the layers is consistent
    # with either the input data dimensions or the output dimensions of the previous layer.

    # This test is to ensure that the modification has been applied at all, not to
    # check the modification itself is exactly what it was in the dictionary

    # if 3D isnt in the model_instance string
    if "3D" not in model_fixture:
        model_fixture = request.getfixturevalue(model_fixture)

        original_model = model_fixture

        modified_model = model_modifier.modify_model_architecture(
            model_fixture,
            correct_modifications_2D,
        )

        assert modified_model is not None

        # Ensure that the model architecture has been modified as expected
        for key, modification in correct_modifications_2D.get(model_name, {}).items():
            attr_name = key.split(".")[-1]
            nested_attr = model_modifier.get_nested_attr(modified_model, key)

            if hasattr(nested_attr, attr_name):
                assert getattr(nested_attr, attr_name) == modification
            else:
                assert nested_attr == modification

        # Ensure that the final prediction layer has been modified as expected but the output dim
        # has not
        if hasattr(original_model, "final_prediction"):
            assert (
                    modified_model.final_prediction[-1].out_features
                    == original_model.final_prediction[-1].out_features
            )


# testing correct modifications: tabular-only or 3D
@pytest.mark.parametrize("model_name, model_fixture", model_instances_training)
def test_correct_modify_model_architecture_3D_training(
        model_name, model_fixture, request
):
    # "modification" may have been modified (ironically) by the calc_fused_layers method
    # in some of the models. This is to ensure that the input to the layers is consistent
    # with either the input data dimensions or the output dimensions of the previous layer.

    # This test is to ensure that the modification has been applied at all, not to
    # check the modification itself is exactly what it was in the dictionary

    # if 3D isnt in the model_instance string
    if "2D" not in model_fixture:
        model_fixture = request.getfixturevalue(model_fixture)

        original_model = model_fixture

        modified_model = model_modifier.modify_model_architecture(
            model_fixture,
            correct_modifications_3D,
        )

        assert modified_model is not None

        # Ensure that the model architecture has been modified as expected
        for key, modification in correct_modifications_3D.get(model_name, {}).items():
            attr_name = key.split(".")[-1]
            nested_attr = model_modifier.get_nested_attr(modified_model, key)

            if hasattr(nested_attr, attr_name):
                assert getattr(nested_attr, attr_name) == modification
            else:
                assert nested_attr == modification

        # Ensure that the final prediction layer has been modified as expected but the output dim has not
        if hasattr(original_model, "final_prediction"):
            assert (
                    modified_model.final_prediction[-1].out_features
                    == original_model.final_prediction[-1].out_features
            )


# Test the modify_model_architecture function with incorrect data type for the modifications
@pytest.mark.parametrize("model_name, model_fixture", model_instances_training)
def test_wrong_data_type_modify_model_architecture_training(
        model_name, model_fixture, request
):
    # iterate through the modifications to check each throws an error
    for key, modification in wrong_layer_type_modifications.get(model_name, {}).items():
        individual_modification = {model_name: {key: modification}}

        # Modify the model's architecture using the function
        with pytest.raises(
                TypeError, match="Incorrect data type for the modifications"
        ):
            model_modifier.modify_model_architecture(
                request.getfixturevalue(model_fixture),
                individual_modification,
            )


# Test the modify_model_architecture function with 3D conv layers with 2D data
@pytest.mark.parametrize("model_name, model_fixture", model_instances_training)
def test_wrong_img_dim_2D_modify_model_architecture_training(
        model_name, model_fixture, request
):
    if "2D" in model_fixture:
        # using correct 3D modifications, which are incorrect for 2D images
        for key, modification in correct_modifications_3D.get(model_name, {}).items():
            individual_modification = {model_name: {key: modification}}

            # Modify the model's architecture using the function
            with pytest.raises(TypeError, match="Incorrect conv layer type"):
                model_modifier.modify_model_architecture(
                    request.getfixturevalue(model_fixture),
                    individual_modification,
                )


# Test the modify_model_architecture function with 2D conv layers with 3D data
@pytest.mark.parametrize("model_name, model_fixture", model_instances_training)
def test_wrong_img_dim_3D_modify_model_architecture_training(
        model_name, model_fixture, request
):
    if "3D" in model_fixture:
        # using correct 3D modifications, which are incorrect for 2D images
        for key, modification in correct_modifications_2D.get(model_name, {}).items():
            individual_modification = {model_name: {key: modification}}

            # Modify the model's architecture using the function
            with pytest.raises(TypeError, match="Incorrect conv layer type"):
                model_modifier.modify_model_architecture(
                    request.getfixturevalue(model_fixture),
                    individual_modification,
                )


######################
# Subspace-creation methods
######################


@pytest.fixture
def model_instance_denoising_autoencoder_subspace_method_2D(create_test_files):
    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    image_torch_file_2d = create_test_files["image_torch_file_2d"]

    sources = [tabular1_csv, tabular2_csv, image_torch_file_2d]
    batch_size = 8
    example_fusion_model = Mock()
    example_fusion_model.modality_type = "tabular_image"

    # Initialize the TrainTestDataModule
    dm = TrainTestDataModule(example_fusion_model,
                             sources,
                             output_paths={"checkpoints": None},
                             prediction_task="binary",
                             batch_size=batch_size,
                             test_size=0.2,
                             multiclass_dimensions=None, )
    dm.prepare_data()
    dm.setup()

    return denoising_autoencoder_subspace_method(datamodule=dm)


@pytest.fixture
def model_instance_denoising_autoencoder_subspace_method_3D(create_test_files):
    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    image_torch_file_3d = create_test_files["image_torch_file_3d"]

    sources = [tabular1_csv, tabular2_csv, image_torch_file_3d]
    batch_size = 8
    example_fusion_model = Mock()
    example_fusion_model.modality_type = "tabular_image"

    # Initialize the TrainTestDataModule
    dm = TrainTestDataModule(example_fusion_model,
                             sources,
                             output_paths={"checkpoints": None},
                             prediction_task="binary",
                             batch_size=batch_size,
                             test_size=0.2,
                             multiclass_dimensions=None, )
    dm.prepare_data()
    dm.setup()

    return denoising_autoencoder_subspace_method(datamodule=dm)


@pytest.fixture
def model_instance_concat_img_latent_tab_subspace_method_2D(create_test_files):
    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    image_torch_file_2d = create_test_files["image_torch_file_2d"]

    sources = [tabular1_csv, tabular2_csv, image_torch_file_2d]
    batch_size = 8
    example_fusion_model = Mock()
    example_fusion_model.modality_type = "tabular_image"

    # Initialize the TrainTestDataModule
    dm = TrainTestDataModule(example_fusion_model,
                             sources,
                             output_paths={"checkpoints": None},
                             prediction_task="binary",
                             batch_size=batch_size,
                             test_size=0.2,
                             multiclass_dimensions=None, )
    dm.prepare_data()
    dm.setup()

    return concat_img_latent_tab_subspace_method(dm)


@pytest.fixture
def model_instance_concat_img_latent_tab_subspace_method_3D(create_test_files):
    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    image_torch_file_3d = create_test_files["image_torch_file_3d"]

    sources = [tabular1_csv, tabular2_csv, image_torch_file_3d]
    batch_size = 8
    example_fusion_model = Mock()
    example_fusion_model.modality_type = "tabular_image"

    # Initialize the TrainTestDataModule
    datamodule = TrainTestDataModule(example_fusion_model,
                                     sources,
                                     output_paths={"checkpoints": None},
                                     prediction_task="binary",
                                     batch_size=batch_size,
                                     test_size=0.2,
                                     multiclass_dimensions=None, )
    datamodule.prepare_data()
    datamodule.setup()

    return concat_img_latent_tab_subspace_method(datamodule=datamodule)


# @pytest.fixture
# def model_instance_MCVAESubspaceMethod(create_test_files):
#     tabular1_csv = create_test_files["tabular1_csv"]
#     tabular2_csv = create_test_files["tabular2_csv"]
#     image_torch_file_3d = create_test_files["image_torch_file_3d"]
#
#     params = {
#         "test_size": 0.2,
#         "prediction_task": "binary",
#         "multiclass_dimensions": None,
#         "checkpoint_dir": None,
#     }
#     sources = [tabular1_csv, tabular2_csv, image_torch_file_3d]
#     batch_size = 8
#     example_fusion_model = Mock()
#     example_fusion_model.modality_type = "tabular_tabular"
#
#     # Initialize the TrainTestDataModule
#     datamodule = TrainTestDataModule(params, example_fusion_model, sources, batch_size)
#     datamodule.prepare_data()
#     datamodule.setup()
#
#     return MCVAESubspaceMethod(datamodule=datamodule)


model_instances_data = [
    (
        "denoising_autoencoder_subspace_method",
        "model_instance_denoising_autoencoder_subspace_method_2D",
    ),
    (
        "denoising_autoencoder_subspace_method",
        "model_instance_denoising_autoencoder_subspace_method_3D",
    ),
    (
        "concat_img_latent_tab_subspace_method",
        "model_instance_concat_img_latent_tab_subspace_method_2D",
    ),
    (
        "concat_img_latent_tab_subspace_method",
        "model_instance_concat_img_latent_tab_subspace_method_3D",
    ),
    # ("MCVAESubspaceMethod", "model_instance_MCVAESubspaceMethod"),
]


# testing correct modifications: tabular-only or 2D
@pytest.mark.parametrize("model_name, model_fixture", model_instances_data)
def test_correct_modify_model_architecture_2D_data(model_name, model_fixture, request):
    # if 3D isnt in the model_instance string
    if "3D" not in model_fixture:
        model_fixture = request.getfixturevalue(model_fixture)

        original_model = model_fixture

        modified_model = model_modifier.modify_model_architecture(
            model_fixture, correct_modifications_2D
        )

        assert modified_model is not None

        # Ensure that the model architecture has been modified as expected
        for key, modification in correct_modifications_2D.get(model_name, {}).items():
            attr_name = key.split(".")[-1]
            nested_attr = model_modifier.get_nested_attr(modified_model, key)

            if hasattr(nested_attr, attr_name):
                assert getattr(nested_attr, attr_name) == modification
            else:
                assert nested_attr == modification

        # Ensure that the final prediction layer has been modified as expected but the output dim has not
        if hasattr(original_model, "final_prediction"):
            assert (
                    modified_model.final_prediction[-1].out_features
                    == original_model.final_prediction[-1].out_features
            )


# testing correct modifications: tabular-only or 3D
@pytest.mark.parametrize("model_name, model_fixture", model_instances_data)
def test_correct_modify_model_architecture_3D_data(model_name, model_fixture, request):
    # if 3D isnt in the model_instance string
    if "2D" not in model_fixture:
        model_fixture = request.getfixturevalue(model_fixture)

        original_model = model_fixture

        modified_model = model_modifier.modify_model_architecture(
            model_fixture, correct_modifications_3D
        )

        assert modified_model is not None

        # Ensure that the model architecture has been modified as expected
        for key, modification in correct_modifications_3D.get(model_name, {}).items():
            attr_name = key.split(".")[-1]
            nested_attr = model_modifier.get_nested_attr(modified_model, key)

            if hasattr(nested_attr, attr_name):
                assert getattr(nested_attr, attr_name) == modification
            else:
                assert nested_attr == modification

        # Ensure that the final prediction layer has been modified as expected but the output dim has not
        if hasattr(original_model, "final_prediction"):
            assert (
                    modified_model.final_prediction[-1].out_features
                    == original_model.final_prediction[-1].out_features
            )


# Test the modify_model_architecture function with incorrect data type for the modifications
@pytest.mark.parametrize("model_name, model_fixture", model_instances_data)
def test_wrong_data_type_modify_model_architecture_data(
        model_name, model_fixture, request
):
    for key, modification in wrong_layer_type_modifications.get(model_name, {}).items():
        individual_modification = {model_name: {key: modification}}

        # Modify the model's architecture using the function
        with pytest.raises(
                TypeError, match="Incorrect data type for the modifications"
        ):
            model_modifier.modify_model_architecture(
                request.getfixturevalue(model_fixture),
                individual_modification,
            )


# Test the modify_model_architecture function with 3D conv layers with 2D data
@pytest.mark.parametrize("model_name, model_fixture", model_instances_data)
def test_wrong_img_dim_2D_modify_model_architecture_data(
        model_name, model_fixture, request
):
    if "2D" in model_fixture:
        # using correct 3D modifications, which are incorrect for 2D images
        listed_dict = [correct_modifications_3D[k] for k in ("all", model_name)]
        new_dict = {"all": listed_dict[0], model_name: listed_dict[1]}

        # only test whether the error is raised if we are actually modifying the conv layers
        conv_layer_found = any(
            "Conv" in str(value2)
            for value in new_dict.values()
            for value2 in value.values()
        )

        if conv_layer_found:
            # Modify the model's architecture using the function
            with pytest.raises(TypeError, match="Incorrect conv layer type"):
                model_modifier.modify_model_architecture(
                    request.getfixturevalue(model_fixture),
                    new_dict,
                )


# Test the modify_model_architecture function with 2D conv layers with 3D data
@pytest.mark.parametrize("model_name, model_fixture", model_instances_data)
def test_wrong_img_dim_3D_modify_model_architecture_data(
        model_name, model_fixture, request
):
    if "3D" in model_fixture:
        listed_dict = [correct_modifications_2D[k] for k in ("all", model_name)]
        new_dict = {"all": listed_dict[0], model_name: listed_dict[1]}

        # only test whether the error is raised if we are actually modifying the conv layers
        conv_layer_found = any(
            "Conv" in str(value2)
            for value in new_dict.values()
            for value2 in value.values()
        )

        if conv_layer_found:
            # Modify the model's architecture using the function
            with pytest.raises(TypeError, match="Incorrect conv layer type"):
                model_modifier.modify_model_architecture(
                    request.getfixturevalue(model_fixture),
                    new_dict,
                )
