import pytest
import torch.nn as nn

# from fusilli.train import modify_model_architecture, get_nested_attr
from fusilli.utils import model_modifier

from fusilli.fusionmodels.tabularimagefusion.concat_img_maps_tabular_data import (
    ConcatImageMapsTabularData,
)
from fusilli.fusionmodels.tabularimagefusion.concat_img_maps_tabular_maps import (
    ConcatImageMapsTabularMaps,
)
from fusilli.fusionmodels.tabularfusion.concat_data import (
    ConcatTabularData,
)
from fusilli.fusionmodels.tabularfusion.concat_feature_maps import (
    ConcatTabularFeatureMaps,
)
from fusilli.fusionmodels.tabularimagefusion.decision import ImageDecision
from fusilli.fusionmodels.tabularfusion.decision import TabularDecision

from fusilli.fusionmodels.tabularfusion.activation import ActivationFusion
from fusilli.fusionmodels.tabularfusion.attention_and_activation import AttentionAndSelfActivation

import torch

correct_modifications_2D = {
    "ConcatImageMapsTabularData": {
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
            nn.Linear(25, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
    },
    "ConcatImageMapsTabularMaps": {
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
            nn.Linear(25, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
    },
    "ConcatTabularData": {
        "fused_layers": nn.Sequential(
            nn.Linear(25, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
    },
    "ConcatTabularFeatureMaps": {
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
        "mod2_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(20, 45),
                    nn.ReLU(),
                ),
                "layer 2": nn.Sequential(
                    nn.Linear(45, 70),
                    nn.ReLU(),
                ),
                "layer 3": nn.Sequential(
                    nn.Linear(70, 100),
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
    "ImageDecision": {
        "fusion_operation": lambda x: torch.mean(x, dim=1),
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
    "TabularDecision": {
        "fusion_operation": lambda x: torch.mean(x, dim=1),
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
        "mod2_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(20, 45),
                    nn.ReLU(),
                ),
                "layer 2": nn.Sequential(
                    nn.Linear(45, 70),
                    nn.ReLU(),
                ),
                "layer 3": nn.Sequential(
                    nn.Linear(70, 100),
                    nn.ReLU(),
                ),
            }
        ),
    },
    "ActivationFusion": {
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
        "mod2_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(20, 45),
                    nn.ReLU(),
                ),
                "layer 2": nn.Sequential(
                    nn.Linear(45, 70),
                    nn.ReLU(),
                ),
                "layer 3": nn.Sequential(
                    nn.Linear(70, 128),
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
    "AttentionAndSelfActivation": {
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
        "mod2_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(20, 45),
                    nn.ReLU(),
                ),
                "layer 2": nn.Sequential(
                    nn.Linear(45, 70),
                    nn.ReLU(),
                ),
                "layer 3": nn.Sequential(
                    nn.Linear(70, 128),
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
        "attention_reduction_ratio": 2,
    }
}

correct_modifications_3D = {
    "ConcatImageMapsTabularData": {
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
            nn.Linear(25, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
    },
    "ConcatImageMapsTabularMaps": {
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
            nn.Linear(25, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
    },
    "ConcatTabularData": {
        "fused_layers": nn.Sequential(
            nn.Linear(25, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
    },
    "ConcatTabularFeatureMaps": {
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
        "mod2_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(20, 45),
                    nn.ReLU(),
                ),
                "layer 2": nn.Sequential(
                    nn.Linear(45, 70),
                    nn.ReLU(),
                ),
                "layer 3": nn.Sequential(
                    nn.Linear(70, 100),
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
    "ImageDecision": {
        "fusion_operation": lambda x: torch.mean(x, dim=1),
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
    "TabularDecision": {
        "fusion_operation": lambda x: torch.mean(x, dim=1),
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
        "mod2_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(20, 45),
                    nn.ReLU(),
                ),
                "layer 2": nn.Sequential(
                    nn.Linear(45, 70),
                    nn.ReLU(),
                ),
                "layer 3": nn.Sequential(
                    nn.Linear(70, 100),
                    nn.ReLU(),
                ),
            }
        ),
    },
}

wrong_dtype_modifications = {
    "ConcatImageMapsTabularData": {
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
    "ConcatImageMapsTabularMaps": {
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
    "ConcatTabularData": {
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
    "ConcatTabularFeatureMaps": {
        "mod1_layers": nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 66),
            nn.ReLU(),
            nn.Linear(66, 128),
            nn.ReLU(),
        ),
        "mod2_layers": nn.Sequential(
            nn.Linear(20, 45),
            nn.ReLU(),
            nn.Linear(45, 70),
            nn.ReLU(),
            nn.Linear(70, 100),
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
    "ImageDecision": {
        "fusion_operation": "median",
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
    "TabularDecision": {
        "fusion_operation": "median",
        "mod1_layers": nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 66),
            nn.ReLU(),
            nn.Linear(66, 128),
            nn.ReLU(),
        ),
        "mod2_layers": nn.Sequential(
            nn.Linear(20, 45),
            nn.ReLU(),
            nn.Linear(45, 70),
            nn.ReLU(),
            nn.Linear(70, 100),
            nn.ReLU(),
        ),
    },
    "ActivationFusion": {
        "mod1_layers": nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 66),
            nn.ReLU(),
            nn.Linear(66, 128),
            nn.ReLU(),
        ),
        "mod2_layers": nn.Sequential(
            nn.Linear(20, 45),
            nn.ReLU(),
            nn.Linear(45, 70),
            nn.ReLU(),
            nn.Linear(70, 100),
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
    "AttentionAndSelfActivation": {
        "mod1_layers": nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 66),
            nn.ReLU(),
            nn.Linear(66, 128),
            nn.ReLU(),
        ),
        "mod2_layers": nn.Sequential(
            nn.Linear(20, 45),
            nn.ReLU(),
            nn.Linear(45, 70),
            nn.ReLU(),
            nn.Linear(70, 100),
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
        "attention_reduction_ratio": "two",
    }
}


@pytest.fixture
def model_instance_ConcatImageMapsTabularData_2D():
    pred_type = "regression"
    data_dims = [10, None, [100, 100]]
    params = {}
    return ConcatImageMapsTabularData(pred_type, data_dims, params)


@pytest.fixture
def model_instance_ConcatImageMapsTabularData_3D():
    pred_type = "regression"
    data_dims = [10, None, [100, 100, 100]]
    params = {}
    return ConcatImageMapsTabularData(pred_type, data_dims, params)


@pytest.fixture
def model_instance_ConcatImageMapsTabularMaps_2D():
    pred_type = "regression"
    data_dims = [10, None, [100, 100]]
    params = {}
    return ConcatImageMapsTabularMaps(pred_type, data_dims, params)


@pytest.fixture
def model_instance_ConcatImageMapsTabularMaps_3D():
    pred_type = "regression"
    data_dims = [10, None, [100, 100, 100]]
    params = {}
    return ConcatImageMapsTabularMaps(pred_type, data_dims, params)


@pytest.fixture
def model_instance_ImageDecision_2D():
    pred_type = "regression"
    data_dims = [10, None, [100, 100]]
    params = {}
    return ImageDecision(pred_type, data_dims, params)


@pytest.fixture
def model_instance_ImageDecision_3D():
    pred_type = "regression"
    data_dims = [10, None, [100, 100, 100]]
    params = {}
    return ImageDecision(pred_type, data_dims, params)


@pytest.fixture
def model_instance_TabularDecision():
    pred_type = "regression"
    data_dims = [10, 15, None]
    params = {}
    return TabularDecision(pred_type, data_dims, params)


@pytest.fixture
def model_instance_ConcatTabularData():
    pred_type = "regression"
    data_dims = [10, 15, None]
    params = {}
    return ConcatTabularData(pred_type, data_dims, params)


@pytest.fixture
def model_instance_ConcatTabularFeatureMaps():
    pred_type = "regression"
    data_dims = [10, 15, None]
    params = {}
    return ConcatTabularFeatureMaps(pred_type, data_dims, params)


@pytest.fixture
def model_instance_ActivationFusion():
    pred_type = "regression"
    data_dims = [10, 15, None]
    params = {}
    return ActivationFusion(pred_type, data_dims, params)


@pytest.fixture
def model_instance_AttentionAndSelfActivation():
    pred_type = "regression"
    data_dims = [10, 15, None]
    params = {}
    return AttentionAndSelfActivation(pred_type, data_dims, params)


model_instances = [
    ("ConcatImageMapsTabularData", "model_instance_ConcatImageMapsTabularData_2D"),
    ("ConcatImageMapsTabularData", "model_instance_ConcatImageMapsTabularData_3D"),
    ("ConcatImageMapsTabularMaps", "model_instance_ConcatImageMapsTabularMaps_2D"),
    ("ConcatImageMapsTabularMaps", "model_instance_ConcatImageMapsTabularMaps_3D"),
    ("ImageDecision", "model_instance_ImageDecision_2D"),
    ("ImageDecision", "model_instance_ImageDecision_3D"),
    ("TabularDecision", "model_instance_TabularDecision"),
    ("ConcatTabularData", "model_instance_ConcatTabularData"),
    ("ConcatTabularFeatureMaps", "model_instance_ConcatTabularFeatureMaps"),
    ("ActivationFusion", "model_instance_ActivationFusion"),
    ("AttentionAndSelfActivation", "model_instance_AttentionAndSelfActivation"),
]


# Test the modify_model_architecture function
@pytest.mark.parametrize("model_name, model_fixture", model_instances)
def test_correct_modify_model_architecture(model_name, model_fixture, request):
    # "modification" may have been modified (ironically) by the calc_fused_layers method
    # in some of the models. This is to ensure that the input to the layers is consistent
    # with either the input data dimensions or the output dimensions of the previous layer.

    # This test is to ensure that the modification has been applied at all, not to
    # check the modification itself is exactly what it was in the dictionary

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
            assert getattr(modified_model, key) == modification

    # Ensure that the final prediction layer has been modified as expected but the output dim
    # has not
    assert (
            modified_model.final_prediction[-1].out_features
            == original_model.final_prediction[-1].out_features
    )


# test incorrect data types


# Test the modify_model_architecture function with incorrect data type for the modifications
@pytest.mark.parametrize("model_name, model_fixture", model_instances)
def test_wrong_data_type_modify_model_architecture_training(
        model_name, model_fixture, request
):
    # iterate through the modifications to check each throws an error
    for key, modification in wrong_dtype_modifications.get(model_name, {}).items():
        individual_modification = {model_name: {key: modification}}
        # Modify the model's architecture using the function
        print(individual_modification)
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
