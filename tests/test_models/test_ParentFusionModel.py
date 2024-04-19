import pytest

from fusilli.fusionmodels.base_model import ParentFusionModel

import torch
import torch.nn as nn


class SampleFusionModel(ParentFusionModel, nn.Module):

    def __init__(self, prediction_task, data_dims, multiclass_dimensions):
        ParentFusionModel.__init__(
            self, prediction_task, data_dims, multiclass_dimensions
        )

    def forward(self, x):
        return [
            torch.rand((x.shape[0], 1)),
        ]


# testing the correct final prediction layer is set for each pred_type
@pytest.mark.parametrize(
    "prediction_task, data_dims, multiclass_dimensions, expected",
    [
        (
            "binary",
            {"mod1_dim": 10, "mod2_dim": 11, "mod3_dim": 12, "img_dim": (100, 100)},
            None,
            nn.Sequential(nn.Linear(100, 1), nn.Sigmoid()),
        ),
        (
            "multiclass",
            {"mod1_dim": 10, "mod2_dim": 11, "mod3_dim": 12, "img_dim": (100, 100)},
            3,
            nn.Sequential(nn.Linear(100, 3)),
        ),
        (
            "regression",
            {"mod1_dim": 10, "mod2_dim": 11, "mod3_dim": 12, "img_dim": (100, 100)},
            None,
            nn.Sequential(nn.Linear(100, 1)),
        ),
    ],
)
def test_set_final_pred_layers(
    prediction_task, data_dims, multiclass_dimensions, expected
):
    model = SampleFusionModel(prediction_task, data_dims, multiclass_dimensions)
    model.set_final_pred_layers(100)
    assert model.final_prediction.__str__() == expected.__str__()


def test_set_mod_1_layers():
    data_dims = {"mod1_dim": 10, "mod2_dim": 11, "mod3_dim": 12, "img_dim": (100, 100)}
    model = SampleFusionModel("binary", data_dims, multiclass_dimensions=None)
    model.set_mod1_layers()
    assert model.mod1_layers["layer 1"][0].in_features == 10


def test_set_mod_2_layers():
    data_dims = {"mod1_dim": 10, "mod2_dim": 11, "mod3_dim": 12, "img_dim": (100, 100)}
    model = SampleFusionModel("binary", data_dims, multiclass_dimensions=None)
    model.set_mod2_layers()
    assert model.mod2_layers["layer 1"][0].in_features == 11


def test_set_mod_3_layers():
    data_dims = {"mod1_dim": 10, "mod2_dim": 11, "mod3_dim": 12, "img_dim": (100, 100)}
    model = SampleFusionModel("binary", data_dims, multiclass_dimensions=None)
    model.set_mod3_layers()
    assert model.mod3_layers["layer 1"][0].in_features == 12


def test_set_img_layers():
    # 2D
    data_dims = {"mod1_dim": 10, "mod2_dim": 11, "mod3_dim": 12, "img_dim": (100, 100)}
    model = SampleFusionModel("binary", data_dims, None)
    model.set_img_layers()
    assert "Conv2d" in str(model.img_layers["layer 1"][0])

    # 3D
    data_dims = {
        "mod1_dim": 10,
        "mod2_dim": 11,
        "mod3_dim": 12,
        "img_dim": (100, 100, 100),
    }
    model = SampleFusionModel("binary", data_dims, None)
    model.set_img_layers()
    assert "Conv3d" in str(model.img_layers["layer 1"][0])


def test_set_fused_layers():
    model = SampleFusionModel(
        "binary",
        {"mod1_dim": 10, "mod2_dim": 11, "mod3_dim": 12, "img_dim": (100, 100)},
        None,
    )
    model.set_fused_layers(250)
    assert model.fused_layers[0].in_features == 250
