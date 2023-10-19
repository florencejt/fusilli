import pytest
import torch.nn as nn
from fusilli.utils import model_modifier
from fusilli.fusionmodels.unimodal.image import ImgUnimodal
from fusilli.fusionmodels.unimodal.tabular1 import Tabular1Unimodal
from fusilli.fusionmodels.unimodal.tabular2 import Tabular2Unimodal


@pytest.fixture
def model_instance_ImgUnimodal_3D():
    pred_type = "regression"
    data_dims = [None, None, [100, 100, 100]]
    params = {}
    return ImgUnimodal(pred_type, data_dims, params)


@pytest.fixture
def model_instance_ImgUnimodal_2D():
    pred_type = "regression"
    data_dims = [None, None, [100, 100]]
    params = {}
    return ImgUnimodal(pred_type, data_dims, params)


@pytest.fixture
def model_instance_Tabular1Unimodal():
    pred_type = "regression"
    data_dims = [10, None, None]
    params = {}
    return Tabular1Unimodal(pred_type, data_dims, params)


@pytest.fixture
def model_instance_Tabular2Unimodal():
    pred_type = "regression"
    data_dims = [None, 15, None]
    params = {}
    return Tabular2Unimodal(pred_type, data_dims, params)


correct_modifications_2D = {
    "ImgUnimodal": {
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
        "fused_layers": nn.Sequential(
            nn.Linear(30, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
    },
    "Tabular1Unimodal": {
        "mod1_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(10, 32),
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
        "fused_layers": nn.Sequential(
            nn.Linear(80, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
    },
    "Tabular2Unimodal": {
        "mod2_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(15, 32),
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

correct_modifications_3D = {
    "ImgUnimodal": {
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
        "fused_layers": nn.Sequential(
            nn.Linear(30, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
    }
}

incorrect_dtype_modifications = {
    "ImgUnimodal": {
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
    "Tabular1Unimodal": {
        "mod1_layers": nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 66),
            nn.ReLU(),
            nn.Linear(66, 128),
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
    "Tabular2Unimodal": {
        "mod2_layers": nn.Sequential(
            nn.Linear(15, 32),
            nn.ReLU(),
            nn.Linear(32, 66),
            nn.ReLU(),
            nn.Linear(66, 128),
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
}

model_instances = [
    ("ImgUnimodal", "model_instance_ImgUnimodal_2D"),
    ("ImgUnimodal", "model_instance_ImgUnimodal_3D"),
    ("Tabular1Unimodal", "model_instance_Tabular1Unimodal"),
    ("Tabular2Unimodal", "model_instance_Tabular2Unimodal"),
]


@pytest.mark.parametrize("model_name, model_fixture", model_instances)
def test_correct_modify_model_architecture(model_name, model_fixture, request):
    if "3D" in model_fixture:
        correct_modifications = correct_modifications_3D
    else:
        correct_modifications = correct_modifications_2D

    # if "3D" not in model_fixture:  # not including the 3D images for this test
    model_fixture = request.getfixturevalue(model_fixture)

    original_model = model_fixture

    modified_model = model_modifier.modify_model_architecture(
        model_fixture, correct_modifications
    )

    # Ensure that the model architecture has been modified as expected
    for key, modification in correct_modifications.get(model_name, {}).items():
        if hasattr(getattr(modified_model, key), "__code__"):
            assert (
                    getattr(modified_model, key).__code__.co_code
                    == modification.__code__.co_code
            )
        else:
            # "modification" may have been modified (ironically) by the calc_fused_layers method
            # in some of the models. This is to ensure that the input to the layers is consistent
            # with either the input data dimensions or the output dimensions of the previous layer.

            # This test is to ensure that the modification has been applied at all, not to
            # check the modification itself is exactly what it was in the dictionary
            assert getattr(modified_model, key) == modification

    # Ensure that the final prediction layer has been modified as expected but the output dim
    # has not
    assert (
            modified_model.final_prediction[-1].out_features
            == original_model.final_prediction[-1].out_features
    )


# Test the modify_model_architecture function with incorrect data type for the modifications
@pytest.mark.parametrize("model_name, model_fixture", model_instances)
def test_wrong_data_type_modify_model_architecture_training(
        model_name, model_fixture, request
):
    # iterate through the modifications to check each throws an error
    for key, modification in incorrect_dtype_modifications.get(model_name, {}).items():
        individual_modification = {model_name: {key: modification}}
        # Modify the model's architecture using the function

        with pytest.raises(
                TypeError, match="Incorrect data type for the modifications"
        ):
            model_modifier.modify_model_architecture(
                request.getfixturevalue(model_fixture),
                individual_modification,
            )


# test incorrect img dimensions
# Test the modify_model_architecture function with 3D conv layers with 2D data
@pytest.mark.parametrize("model_name, model_fixture", model_instances)
def test_wrong_img_dim_2D_modify_model_architecture_data(
        model_name, model_fixture, request
):
    if "2D" in model_fixture:
        # using correct 3D modifications, which are incorrect for 2D images
        listed_dict = correct_modifications_3D[model_name]
        new_dict = {model_name: listed_dict}

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
@pytest.mark.parametrize("model_name, model_fixture", model_instances)
def test_wrong_img_dim_3D_modify_model_architecture_data(
        model_name, model_fixture, request
):
    if "3D" in model_fixture:
        listed_dict = correct_modifications_2D[model_name]
        new_dict = {model_name: listed_dict}

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
