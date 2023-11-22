import pytest
import torch
import torch.nn as nn

from fusilli.utils.model_chooser import import_chosen_fusion_models
from fusilli.fusionmodels.base_model import ParentFusionModel

fusion_models = import_chosen_fusion_models({
    "fusion_type": "unimodal"
}, skip_models=["MCVAE_tab"])

fusion_model_names = [model.__name__ for model in fusion_models]
# make into dict
fusion_model_dict = {fusion_model_names[i]: fusion_models[i] for i in range(len(fusion_model_names))}
print(fusion_model_dict)


def test_Tabular1Unimodal():
    test_model = fusion_model_dict["Tabular1Unimodal"]

    # attributes before initialising
    assert hasattr(test_model, "method_name")
    assert test_model.method_name == "Tabular1 uni-modal"
    assert hasattr(test_model, "modality_type")
    assert test_model.modality_type == "tabular1"
    assert hasattr(test_model, "fusion_type")
    assert test_model.fusion_type == "unimodal"

    # attributes after initialising
    test_model = test_model("binary", [10, None, None], {})

    assert isinstance(test_model, ParentFusionModel)
    assert isinstance(test_model, nn.Module)
    assert hasattr(test_model, "mod1_layers")
    assert test_model.mod1_layers['layer 1'][0].in_features == 10
    assert hasattr(test_model, "fused_dim")
    assert test_model.fused_dim == test_model.mod1_layers['layer 5'][0].out_features
    assert hasattr(test_model, "fused_layers")
    assert hasattr(test_model, "final_prediction")

    # forward pass
    test_input = torch.randn(8, 10)
    test_output = test_model.forward(test_input)
    assert isinstance(test_output, list)
    assert test_output[0].shape == torch.Size([8, 1])
    assert len(test_output) == 1

    with pytest.raises(TypeError, match=r"Wrong input type for model! Expected torch.Tensor"):
        test_model.forward([torch.randn(8, 25)])


def test_Tabular2Unimodal():
    test_model = fusion_model_dict["Tabular2Unimodal"]

    # attributes before initialising
    assert hasattr(test_model, "method_name")
    assert test_model.method_name == "Tabular2 uni-modal"
    assert hasattr(test_model, "modality_type")
    assert test_model.modality_type == "tabular2"
    assert hasattr(test_model, "fusion_type")
    assert test_model.fusion_type == "unimodal"

    # attributes after initialising
    test_model = test_model("binary", [None, 15, None], {})

    assert isinstance(test_model, ParentFusionModel)
    assert isinstance(test_model, nn.Module)
    assert hasattr(test_model, "mod2_layers")
    assert test_model.mod2_layers['layer 1'][0].in_features == 15
    assert hasattr(test_model, "fused_dim")
    assert test_model.fused_dim == test_model.mod2_layers['layer 5'][0].out_features
    assert hasattr(test_model, "fused_layers")
    assert hasattr(test_model, "final_prediction")

    # forward pass
    test_input = torch.randn(8, 15)
    test_output = test_model.forward(test_input)
    assert isinstance(test_output, list)
    assert test_output[0].shape == torch.Size([8, 1])
    assert len(test_output) == 1

    with pytest.raises(TypeError, match=r"Wrong input type for model! Expected torch.Tensor"):
        test_model.forward([torch.randn(8, 15)])


def test_ImgUnimodal():
    test_model = fusion_model_dict["ImgUnimodal"]

    # attributes before initialising
    assert hasattr(test_model, "method_name")
    assert test_model.method_name == "Image unimodal"
    assert hasattr(test_model, "modality_type")
    assert test_model.modality_type == "img"
    assert hasattr(test_model, "fusion_type")
    assert test_model.fusion_type == "unimodal"

    # attributes after initialising
    test_model = test_model("binary", [None, None, (100, 100)], {})

    assert isinstance(test_model, ParentFusionModel)
    assert isinstance(test_model, nn.Module)
    assert hasattr(test_model, "img_layers")
    assert hasattr(test_model, "fused_dim")
    assert hasattr(test_model, "fused_layers")
    assert hasattr(test_model, "final_prediction")

    # forward pass
    test_input = torch.rand((1, 1, 100, 100))

    test_output = test_model.forward(test_input)
    assert isinstance(test_output, list)
    assert test_output[0].shape == torch.Size([1, 1])
    assert len(test_output) == 1

    with pytest.raises(TypeError, match=r"Wrong input type for model! Expected torch.Tensor"):
        test_model.forward([torch.randn(8, 15)])
