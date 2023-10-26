""" 
Functions for modifying aspects of the model (e.g. changing layers, latent space sizes, etc.).
"""

import torch.nn as nn
import warnings
from copy import deepcopy


def modify_model_architecture(model, architecture_modification):
    """
    Modify the architecture of a deep learning model based on the provided configuration.

    Parameters
    ----------
    model : nn.Module
        The original deep learning model.
    architecture_modification : dict
        A dictionary containing architecture modifications.
        Input format {"model": {"layer_group": "modification"}, ...}.
        e.g. {"TabularCrossmodalAttention": {"mod1_layers": new mod 1 layers nn.ModuleDict}}
    
    Returns
    -------
    nn.Module
        The modified deep learning model.
    """

    for model_name, layer_groups in architecture_modification.items():
        # Modify layers for all specified models
        if model_name == "all":
            for layer_group, modification in layer_groups.items():
                if hasattr(model, layer_group):
                    setattr(model, layer_group, modification)
                    print("Changed", layer_group, "in", model_name)
                else:
                    warnings.warn(
                        f"Layer group {layer_group} not found in {model} when flagged with\
                        {model_name}"
                    )
            reset_fused_layers(model, model)

        # Modify layers for a specific model class
        elif model_name == model.__class__.__name__:

            nested_attrs = []  # list of nested attributes to modify

            for layer_group, modification in layer_groups.items():
                nested_attr = get_nested_attr(model, layer_group)

                if hasattr(nested_attr, layer_group.split(".")[-1]):

                    setattr(nested_attr, layer_group.split(".")[-1], modification)
                    print("Changed", layer_group.split(".")[-1], "in", model_name)

                else:
                    raise ValueError(
                        f"Layer group {layer_group} not found in {model_name}"
                    )
                if nested_attr not in nested_attrs:
                    nested_attrs.append(nested_attr)
                # if we're on the last layer group, reset the fused layers
                if layer_group == list(layer_groups.keys())[-1]:
                    for nested_attr in nested_attrs:
                        reset_fused_layers(nested_attr, model)
    return model


def get_nested_attr(obj, attr_path):
    """
    Get a nested attribute from an object using dot-separated path.

    Parameters
    ----------
    obj : object
        The object to retrieve the nested attribute from.
    attr_path : str
        Dot-separated path to the nested attribute.
    
    Returns
    -------
    object
        The nested attribute if found, otherwise None.
    """

    attributes = attr_path.split(".")

    if len(attributes) > 1:  # if we're looking for a nested attribute
        attr = getattr(obj, attributes[0])

        # if the attribute is more than one . deep
        for i in range(1, len(attributes) - 1):
            attr = getattr(attr, attributes[i])

    else:
        attr = obj

    return attr


def reset_fused_layers(obj, model):
    """
    Reset fused layers of a model if the reset method is available.

    Parameters
    ----------
    obj : nn.Module
        The model to reset fused layers for.
    model : nn.Module
        The original deep learning model.
    """

    if hasattr(obj, "calc_fused_layers"):
        obj.calc_fused_layers()
        print("Reset fused layers in", model.__class__.__name__)

    if hasattr(model, "check_params"):
        model.check_params()
        print("Checked params in", model.__class__.__name__)
