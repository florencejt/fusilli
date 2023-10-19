"""
This module contains functions to check the validity of the model modifications, such as
checking the data type of the modifications and checking the input image dimensions of the
modifications.
"""

import torch.nn as nn
import torch


def check_dtype(attribute, correct_dtype, attribute_name):
    """Check if the modification is of the correct data type.

    Parameters
    ----------
    attribute : object
        Attribute to check.
    correct_dtype : object
        Correct data type.
    attribute_name : str
        Name of the attribute to check (for the error message)

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
    attribute_name : str
        Name of the attribute to check (for the error message)

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
    else:
        raise TypeError(
            (
                f"Incorrect data type for the modifications: Attribute {attribute_name}"
                f" must be of type nn.ModuleDict or nn.Sequential, not dtype {type(attribute).__name__}.",
            )
        )

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
    attribute_name : str
        Name of the attribute to check (for the error message)

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
    Check that the fused layers within the fusion model (meaning the layers that take place
    after the fusion) are of the correct data type (nn.Sequential) and that the first layer
    is a nn.Linear layer with the correct in_features. Also returns the output dimension of
    the fused layers for the final prediction layers of the model.

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


def check_model_input(x, uni_modal_flag=False, correct_length=2):
    """
    Check that the input to the model is of the correct length.

    Parameters
    ----------
    x : tuple or torch.Tensor
        Input to the model's forward function. Should either be a tuple of length 2 for multi-modal methods or a
        torch tensor for uni-modal methods.
    uni_modal_flag : bool
        Flag to indicate whether the model is uni-modal or multi-modal. If True, the input should be a torch tensor.
        If False, the input should be a tuple of length 2. Default is False.
    correct_length : int
        Correct length of the input to the model if it is multi-modal. Default is 2.
        Could be 3 for the graph methods.
    """

    if uni_modal_flag:
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"Wrong input type for model! Expected torch.Tensor, not {type(x)}."
            )
    else:
        if not isinstance(x, tuple):
            raise TypeError(
                f"Wrong input type for model! Expected tuple, not {type(x)}."
            )
        elif len(x) != correct_length:
            raise ValueError(
                f"Wrong number of inputs for model! Expected {correct_length}, not {len(x)}."
            )
        elif not all(isinstance(x_i, torch.Tensor) for x_i in x):
            raise TypeError(
                f"Wrong input type for model! Expected list of torch.Tensors, not {type(x)}."
            )
