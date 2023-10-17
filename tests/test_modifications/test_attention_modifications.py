import pytest
import torch.nn as nn

from fusilli.fusionmodels.tabularimagefusion.crossmodal_att import (
    CrossmodalMultiheadAttention,
)
from fusilli.fusionmodels.tabularimagefusion.channelwise_att import (
    ImageChannelWiseMultiAttention,
)
from fusilli.fusionmodels.tabularfusion.crossmodal_att import (
    TabularCrossmodalMultiheadAttention,
)
from fusilli.fusionmodels.tabularfusion.channelwise_att import (
    TabularChannelWiseMultiAttention,
)

from fusilli.utils import model_modifier

correct_modifications = {  # Correct modifications
    "TabularChannelWiseMultiAttention": {
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
                    nn.Linear(15, 32),
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
            }
        ),
    },
    "ImageChannelWiseMultiAttention": {
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
    "TabularCrossmodalAttention": {
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
                    nn.Linear(15, 32),
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
            }
        ),
    },
    "CrossmodalMultiheadAttention": {
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
}

wrong_img_dim_modifications_3D = {
    "ImageChannelWiseMultiAttention": {
        "img_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Conv3d(1, 8, kernel_size=(3, 3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool3d((2, 2, 2)),
                ),
                "layer 2": nn.Sequential(
                    nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool3d((2, 2, 2)),
                ),
                "layer 3": nn.Sequential(
                    nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool3d((2, 2, 2)),
                ),
            }
        ),
    },
    "CrossmodalMultiheadAttention": {
        "img_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Conv3d(1, 8, kernel_size=(3, 3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool3d((2, 2, 2)),
                ),
                "layer 2": nn.Sequential(
                    nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool3d((2, 2, 2)),
                ),
                "layer 3": nn.Sequential(
                    nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool3d((2, 2, 2)),
                ),
            }
        ),
    },
}  # 2D image input instead of 3D

wrong_img_dim_modifications_2D = {
    "ImageChannelWiseMultiAttention": {
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
    "CrossmodalMultiheadAttention": {
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
}  # 2D image input instead of 3D

wrong_num_layers_modifications = {
    "TabularChannelWiseMultiAttention": {
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
                    nn.Linear(15, 32),
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
            }
        ),
    },
    "TabularCrossmodalMultiheadAttention": {
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
                    nn.Linear(15, 32),
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
            }
        ),
    },
    "ImageChannelWiseMultiAttention": {
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
                "layer 4": nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
            }
        ),
    },
    "CrossmodalMultiheadAttention": {
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
                "layer 4": nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
            }
        ),
    },
}  # wrong number of layers

wrong_layer_type_modifications = {
    "TabularChannelWiseMultiAttention": {
        "mod1_layers": nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 66),
            nn.ReLU(),
            nn.Linear(66, 128),
            nn.ReLU(),
        ),
        "mod2_layers": nn.Sequential(
            nn.Linear(15, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        ),
    },
    "ImageChannelWiseMultiAttention": {
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
    "TabularCrossmodalMultiheadAttention": {
        "mod1_layers": nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 66),
            nn.ReLU(),
            nn.Linear(66, 128),
            nn.ReLU(),
        ),
        "mod2_layers": nn.Sequential(
            nn.Linear(15, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        ),
    },
    "CrossmodalMultiheadAttention": {
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
}  # wrong layer type


@pytest.fixture
def model_instance_TabularChannelWiseMultiAttention():
    pred_type = "regression"
    data_dims = [10, 15, None]
    params = {}
    return TabularChannelWiseMultiAttention(pred_type, data_dims, params)


@pytest.fixture
def model_instance_ImageChannelWiseMultiAttention_2D():
    pred_type = "regression"
    data_dims = [10, 15, [100, 100]]
    params = {}
    return ImageChannelWiseMultiAttention(pred_type, data_dims, params)


@pytest.fixture
def model_instance_ImageChannelWiseMultiAttention_3D():
    pred_type = "regression"
    data_dims = [10, 15, [100, 100, 100]]
    params = {}
    return ImageChannelWiseMultiAttention(pred_type, data_dims, params)


@pytest.fixture
def model_instance_TabularCrossmodalMultiheadAttention():
    pred_type = "regression"
    data_dims = [10, 15, None]
    params = {}
    return TabularCrossmodalMultiheadAttention(pred_type, data_dims, params)


@pytest.fixture
def model_instance_CrossmodalMultiheadAttention_2D():
    pred_type = "regression"
    data_dims = [10, 15, [100, 100]]
    params = {}
    return CrossmodalMultiheadAttention(pred_type, data_dims, params)


@pytest.fixture
def model_instance_CrossmodalMultiheadAttention_3D():
    pred_type = "regression"
    data_dims = [10, 15, [100, 100, 100]]
    params = {}
    return CrossmodalMultiheadAttention(pred_type, data_dims, params)


model_instances = [
    (
        "TabularChannelWiseMultiAttention",
        "model_instance_TabularChannelWiseMultiAttention",
    ),
    (
        "ImageChannelWiseMultiAttention",
        "model_instance_ImageChannelWiseMultiAttention_2D",
    ),
    (
        "ImageChannelWiseMultiAttention",
        "model_instance_ImageChannelWiseMultiAttention_3D",
    ),
    (
        "TabularCrossmodalMultiheadAttention",
        "model_instance_TabularCrossmodalMultiheadAttention",
    ),
    ("CrossmodalMultiheadAttention", "model_instance_CrossmodalMultiheadAttention_2D"),
    ("CrossmodalMultiheadAttention", "model_instance_CrossmodalMultiheadAttention_3D"),
]


# Test the modify_model_architecture function
@pytest.mark.parametrize("model_name, model_fixture", model_instances)
def test_correct_modify_model_architecture(model_name, model_fixture, request):
    # "modification" may have been modified (ironically) by the calc_fused_layers method
    # in some of the models. This is to ensure that the input to the layers is consistent
    # with either the input data dimensions or the output dimensions of the previous layer.

    # This test is to ensure that the modification has been applied at all, not to
    # check the modification itself is exactly what it was in the dictionary
    if model_fixture not in [
        "model_instance_ImageChannelWiseMultiAttention_3D",
        "model_instance_CrossmodalMultiheadAttention_3D",
    ]:  # not including the 3D images for this test
        model_fixture = request.getfixturevalue(model_fixture)

        original_model = model_fixture

        # Modify the model's architecture using the function
        modified_model = model_modifier.modify_model_architecture(
            model_fixture, correct_modifications
        )

        # Ensure that the model architecture has been modified as expected
        for key, modification in correct_modifications.get(model_name, {}).items():
            assert getattr(modified_model, key) == modification

        # Ensure that the final prediction layer has been modified as expected but the output dim
        # has not
        assert (
            modified_model.final_prediction[-1].out_features
            == original_model.final_prediction[-1].out_features
        )


# Test the modify_model_architecture function with incorrect number of layers
# For attention-based fusion, the number of layers in the two modalities must be the same
@pytest.mark.parametrize("model_name, model_fixture", model_instances)
def test_wrong_num_layers_modify_model_architecture(model_name, model_fixture, request):
    # Modify the model's architecture using the function
    if model_fixture not in [
        "model_instance_ImageChannelWiseMultiAttention_3D",
        "model_instance_CrossmodalMultiheadAttention_3D",
    ]:  # not including the 3D images for this test
        for key, modification in wrong_num_layers_modifications.get(
            model_name, {}
        ).items():
            individual_modification = {model_name: {key: modification}}
            with pytest.raises(ValueError):
                model_modifier.modify_model_architecture(
                    request.getfixturevalue(model_fixture),
                    individual_modification,
                ), "The number of layers in the two modalities must be the same."


# Test the modify_model_architecture function with incorrect data type for the modifications
@pytest.mark.parametrize("model_name, model_fixture", model_instances)
def test_wrong_data_type_modify_model_architecture(model_name, model_fixture, request):
    # Modify the model's architecture using the function
    for key, modification in wrong_layer_type_modifications.get(model_name, {}).items():
        individual_modification = {model_name: {key: modification}}
        with pytest.raises(
            TypeError, match="Incorrect data type for the modifications"
        ):
            model_modifier.modify_model_architecture(
                request.getfixturevalue(model_fixture),
                individual_modification,
            )


# Test the modify_model_architecture function with incorrect img dim for the modifications
@pytest.mark.parametrize("model_name, model_fixture", model_instances)
def test_wrong_img_dim_modify_model_architecture(model_name, model_fixture, request):
    # Modify the model's architecture using the function
    if model_fixture in [
        "model_instance_ImageChannelWiseMultiAttention_2D",
        "model_instance_CrossmodalMultiheadAttention_2D",
    ]:
        for key, modification in wrong_img_dim_modifications_3D.get(
            model_name, {}
        ).items():
            individual_modification = {model_name: {key: modification}}

            with pytest.raises(TypeError, match="Incorrect conv layer type"):
                model_modifier.modify_model_architecture(
                    request.getfixturevalue(model_fixture),
                    individual_modification,
                )

    elif model_fixture in [
        "model_instance_ImageChannelWiseMultiAttention_3D",
        "model_instance_CrossmodalMultiheadAttention_3D",
    ]:
        for key, modification in wrong_img_dim_modifications_2D.get(
            model_name, {}
        ).items():
            individual_modification = {model_name: {key: modification}}

            with pytest.raises(TypeError, match="Incorrect conv layer type"):
                model_modifier.modify_model_architecture(
                    request.getfixturevalue(model_fixture),
                    wrong_img_dim_modifications_2D,
                )


# Run the tests
if __name__ == "__main__":
    pytest.main()
