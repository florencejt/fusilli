# checking that the check_model_validity functions work as expected

import pytest
import torch.nn as nn

from fusilli.utils.check_model_validity import (
    check_dtype,
    check_img_dim,
    check_var_is_function,
    check_fused_layers,
)


# Test check_dtype function
def test_check_dtype():
    with pytest.raises(TypeError) as excinfo:
        check_dtype(42, str, "attribute_name")
    assert (
        excinfo.value.args[0][0]
        == "Incorrect data type for the modifications: Attribute attribute_name must be of type str, not dtype int."
    )


# Test check_img_dim function
def test_check_img_dim():
    # Create a mock object to simulate the nn.Module
    mock_2D_convs = nn.ModuleDict(
        {
            "conv1": nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3), nn.Conv2d(64, 64, kernel_size=3)
            ),
            "conv2": nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3), nn.Conv2d(64, 64, kernel_size=3)
            ),
        }
    )

    mock_3D_convs = nn.ModuleDict(
        {
            "conv1": nn.Sequential(
                nn.Conv3d(3, 64, kernel_size=3), nn.Conv3d(64, 64, kernel_size=3)
            ),
            "conv2": nn.Sequential(
                nn.Conv3d(3, 64, kernel_size=3), nn.Conv3d(64, 64, kernel_size=3)
            ),
        }
    )

    # Test with Conv2D layer and incorrect img_dim
    with pytest.raises(TypeError) as excinfo:
        check_img_dim(mock_2D_convs, [1, 2, 3], "attribute_name")

    assert (
        str(excinfo.value)
        == "Incorrect conv layer type for the modified attribute_name: input image dimensions are [1, 2, 3] and img layers have a Conv2D layer in them."
    )

    # Test with Conv3D layer and incorrect img_dim
    with pytest.raises(TypeError) as excinfo:
        check_img_dim(mock_3D_convs, [1, 2], "attribute_name")
    assert (
        str(excinfo.value)
        == "Incorrect conv layer type for the modified attribute_name: input image dimensions are [1, 2] and img layers have a Conv3D layer in them."
    )


# Test check_var_is_function function
def test_check_var_is_function():
    with pytest.raises(TypeError) as excinfo:
        check_var_is_function(42, "attribute_name")
    assert (
        excinfo.value.args[0][0]
        == "Incorrect data type for the modifications: Attribute attribute_name must be a function, not dtype int."
    )


# Test check_fused_layers function
def test_check_fused_layers():
    # Create a mock nn.Sequential and a fused_dim value
    fused_layers = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 30))
    fused_dim = 10

    result, out_dim = check_fused_layers(fused_layers, fused_dim)

    assert isinstance(result, nn.Sequential)
    assert out_dim == 30
    assert isinstance(result[0], nn.Linear)
    assert result[0].in_features == fused_dim
