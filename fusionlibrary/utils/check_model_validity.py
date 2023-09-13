"""
Check validity of model modifications.
"""

import torch.nn as nn


def check_dtype(attribute, correct_dtype, attribute_name):
    """Check if the modification is of the correct data type.

    Parameters
    ----------
    attribute : object
        Attribute to check.
    correct_dtype : object
        Correct data type.

    Raises
    ------
    TypeError
        If the modification is not of the correct data type.

    """
    if not isinstance(
        attribute,
        correct_dtype,
    ):
        raise TypeError(
            (
                f"Incorrect data type for the modifications: Attribute {attribute_name}"
                f" must be of type {correct_dtype.__name__}, not dtype {type(attribute).__name__}.",
            )
        )


def check_img_dim(attribute, img_dim, attribute_name):
    """Check if the modification img layers are the correct dimension.

    Parameters
    ----------
    attribute : object
        Attribute to check.
    img_dim : object
        Correct img dimensions.

    Raises
    ------
    TypeError
        If the modification is not of the correct data type.

    """
    if isinstance(attribute, nn.ModuleDict):
        has_conv3d_layer = any(
            isinstance(module, nn.Conv3d)
            for _, sequential_module in attribute.items()
            for module in sequential_module.children()
        )
        has_conv2d_layer = any(
            isinstance(module, nn.Conv2d)
            for _, sequential_module in attribute.items()
            for module in sequential_module.children()
        )
    elif isinstance(attribute, nn.Sequential):
        has_conv3d_layer = any(isinstance(module, nn.Conv3d) for module in attribute)
        has_conv2d_layer = any(isinstance(module, nn.Conv2d) for module in attribute)

    if has_conv2d_layer is None and has_conv3d_layer is None:
        raise TypeError(
            (
                f"Incorrect conv layer type for the modified {attribute_name}: "
                f"input image dimensions are {img_dim} and img layers have no Conv2D or Conv3D "
                "layers in them."
            )
        )

    if has_conv2d_layer and len(img_dim) == 3:
        raise TypeError(
            (
                f"Incorrect conv layer type for the modified {attribute_name}: input image "
                f"dimensions are {img_dim} and img layers have a Conv2D layer in them."
            )
        )
    elif has_conv3d_layer and len(img_dim) == 2:
        print(attribute)
        raise TypeError(
            (
                f"Incorrect conv layer type for the modified {attribute_name}: "
                f"input image dimensions are {img_dim} and img layers have a Conv3D layer in them."
            )
        )


def check_var_is_function(attribute, attribute_name):
    """Check if the modification is a function.

    Parameters
    ----------
    attribute : object
        Attribute to check.

    Raises
    ------
    TypeError
        If the modification is not a function.

    """
    if not hasattr(attribute, "__code__"):
        raise TypeError(
            (
                f"Incorrect data type for the modifications: Attribute {attribute_name}"
                f" must be a function, not dtype {type(attribute).__name__}.",
            )
        )


def check_fused_layers(fused_layers, fused_dim):
    """
    Check the fused layers in a fusion model:

    - check fused_layers is nn.Sequential
    - check the input features of the first layer is the fused_dim
    - find the output features of the last layer to make prediction layer
    - check first layer is a linear layer

    Parameters
    ----------
    fused_layers : nn.ModuleDict
        Fused layers of the model.
    fused_dim : int
        Dimension of the fused layers.

    Returns
    -------
    fused_layers : nn.ModuleDict
        Fused layers of the model. First layer is modified to have the correct in_features.
    out_dim : int
        Output dimension of the fused layers.

    """

    check_dtype(fused_layers, nn.Sequential, "fused_layers")

    check_dtype(fused_layers[0], nn.Linear, "first layer in fused_layers")

    for layer in range(len(fused_layers)):
        if isinstance(fused_layers[layer], nn.Linear):
            out_dim = fused_layers[layer].out_features

    # Make sure first in_features is the fused_dim
    fused_layers[0] = nn.Linear(fused_dim, fused_layers[0].out_features)

    return fused_layers, out_dim
