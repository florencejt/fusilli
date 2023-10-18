import pytest

from fusilli.fusionmodels.base_model import ParentFusionModel

import torch
import torch.nn as nn


class SampleFusionModel(ParentFusionModel, nn.Module):

    def __init__(self, pred_type, data_dims, params):
        ParentFusionModel.__init__(self, pred_type, data_dims, params)

    def forward(self, x):
        return [torch.rand((x.shape[0], 1)), ]

# testing the correct final prediction layer is set for each pred_type
@pytest.mark.parametrize(
    "pred_type, data_dims, params, expected",
    [
        (
            "binary",
            [10,10,(100,100)],
            {"multiclass_dims": None},
            nn.Sequential(nn.Linear(100, 1), nn.Sigmoid()),
        ),
        (
            "multiclass",
            [10,10,(100,100)],
            {"multiclass_dims": 3},
            nn.Sequential(
                nn.Linear(100, 3)
            ),
        ),
        (
            "regression",
            [10,10,(100,100)],
            {"multiclass_dims": None},
            nn.Sequential(nn.Linear(100, 1)),
        ),
    ],
)
def test_set_final_pred_layers(pred_type, data_dims, params, expected):
    model = SampleFusionModel(pred_type, data_dims, params)
    model.set_final_pred_layers(100)
    assert model.final_prediction.__str__() == expected.__str__()

def test_set_mod_1_layers():

    model = SampleFusionModel("binary", [10,15,(100,100)], {"multiclass_dims": None})
    model.set_mod1_layers()
    assert model.mod1_layers["layer 1"][0].in_features == 10


def test_set_mod_2_layers():

    model = SampleFusionModel("binary", [10,15,(100,100)], {"multiclass_dims": None})
    model.set_mod2_layers()
    assert model.mod2_layers["layer 1"][0].in_features == 15


def test_set_img_layers():

    # 2D
    model = SampleFusionModel("binary", [10,15,(100,100)], {"multiclass_dims": None})
    model.set_img_layers()
    assert "Conv2d" in str(model.img_layers["layer 1"][0])

    # 3D
    model = SampleFusionModel("binary", [10,15,(100,100,100)], {"multiclass_dims": None})
    model.set_img_layers()
    assert "Conv3d" in str(model.img_layers["layer 1"][0])


def test_set_fused_layers():

    model = SampleFusionModel("binary", [10,15,(100,100)], {"multiclass_dims": None})
    model.set_fused_layers(250)
    assert model.fused_layers[0].in_features == 250
