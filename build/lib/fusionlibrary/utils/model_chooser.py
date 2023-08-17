"""
Function/functions to help user choose model's to run
"""

import importlib
import pandas as pd
import warnings

# all the fusion models' names and paths
fusion_model_dict = [
    {"name": "Tabular1Unimodal", "path": "fusion_models.tabular1_unimodal"},
    {"name": "Tabular2Unimodal", "path": "fusion_models.tabular2_unimodal"},
    {"name": "ImgUnimodal", "path": "fusion_models.img_unimodal"},
    {
        "name": "ConcatTabularFeatureMaps",
        "path": "fusion_models.concat_tabular_feature_maps",
    },
    {
        "name": "ConcatImageMapsTabularData",
        "path": "fusion_models.concat_img_maps_tabular_data",
    },
    {"name": "ConcatTabularData", "path": "fusion_models.concat_tabular_data"},
    {
        "name": "ConcatImageMapsTabularMaps",
        "path": "fusion_models.concat_img_maps_tabular_maps",
    },
    {
        "name": "TabularChannelWiseMultiAttention",
        "path": "fusion_models.tabular_channelwise_att",
    },
    {
        "name": "ImageChannelWiseMultiAttention",
        "path": "fusion_models.img_tab_channelwise_att",
    },
    {"name": "CrossmodalMultiheadAttention", "path": "fusion_models.crossmodal_att"},
    {
        "name": "TabularCrossmodalMultiheadAttention",
        "path": "fusion_models.tab_crossmodal_att",
    },
    {"name": "TabularDecision", "path": "fusion_models.tabular_decision"},
    {"name": "ImageDecision", "path": "fusion_models.img_tab_decision"},
    {"name": "MCVAE_tab", "path": "fusion_models.mcvae_tab"},
    {
        "name": "ConcatImgLatentTabDoubleTrain",
        "path": "fusion_models.concat_img_latent_tab_doubletrain",
    },
    {
        "name": "ConcatImgLatentTabDoubleLoss",
        "path": "fusion_models.concat_img_latent_tab_doubleloss",
    },
    {"name": "EdgeCorrGNN", "path": "fusion_models.edge_corr_gnn"},
]


def model_importer(fusion_model_dict):
    """
    Imports all the fusion models in the fusion_model_dict.

    Parameters
    ----------
    fusion_model_dict : list
        List of dictionaries containing the fusion models' names and paths.

    Returns
    -------
    fusion_models : list
        List of all the fusion models class objects
    """

    fusion_models = []
    for model in fusion_model_dict:
        module_name = model["name"]
        module_path = "fusionlibrary." + model["path"]

        module = importlib.import_module(module_path)
        module_class = getattr(module, module_name)

        fusion_models.append(module_class)

    return fusion_models


def get_models(conditions_dict, fusion_model_dict=fusion_model_dict):
    """
    Options:
    Feature
    - fusion_type
    - modality_type
    - method_name
    - class_name

    Condition
    - "all"
    - fusion_type: "Uni-modal", "operation", "attention", "subspace", "graph", "tensor",
        or list of some of those
    - modality_type: "tab", "img", "both_tab", "both_img", or list of some of those
    - method_name: any method name or list of method names that are actually in the dict
    - class_name: any class name or list of class names that are actually in the dict
    """

    # raise error if condition is not "all" and feature is not one of the options
    valid_features = ["fusion_type", "modality_type", "method_name", "class_name"]
    valid_fusion_types = [
        "Uni-modal",
        "operation",
        "attention",
        "subspace",
        "graph",
        "tensor",
    ]
    valid_modality_types = ["tabular1", "tabular2", "img", "both_tab", "tab_img"]

    fusion_models = model_importer(fusion_model_dict)

    # get model names, fusion types, modality types

    method_names = [
        fusion_models[i].method_name for i, model in enumerate(fusion_model_dict)
    ]
    fusion_types = [
        fusion_models[i].fusion_type for i, model in enumerate(fusion_model_dict)
    ]
    modality_types = [
        fusion_models[i].modality_type for i, model in enumerate(fusion_model_dict)
    ]

    class_names = [fusion_models[i].__name__ for i, model in enumerate(fusion_models)]

    method_paths = [
        "fusionlibrary." + model["path"] for i, model in enumerate(fusion_model_dict)
    ]

    # create a dataframe of all the models
    models_df = pd.DataFrame(
        {
            "method_name": method_names,
            "fusion_type": fusion_types,
            "modality_type": modality_types,
            "class_name": class_names,
            "method_path": method_paths,
        }
    )

    filtered_models = models_df

    for feature, condition in conditions_dict.items():
        if feature not in valid_features:
            raise ValueError("Invalid feature:", feature)

        if feature == "fusion_type":
            if isinstance(condition, list):
                invalid_fusion_types = [
                    ftype for ftype in condition if ftype not in valid_fusion_types
                ]
                if invalid_fusion_types:
                    raise ValueError(
                        "Invalid fusion types for feature",
                        feature,
                        ":",
                        invalid_fusion_types,
                    )
            elif condition != "all" and condition not in valid_fusion_types:
                raise ValueError(
                    "Invalid fusion type for feature",
                    feature,
                    ". Choose from:",
                    valid_fusion_types,
                )

        elif feature == "modality_type":
            if isinstance(condition, list):
                invalid_modality_types = [
                    mtype for mtype in condition if mtype not in valid_modality_types
                ]
                if invalid_modality_types:
                    raise ValueError(
                        "Invalid modality types for feature",
                        feature,
                        ":",
                        invalid_modality_types,
                    )
            elif condition != "all" and condition not in valid_modality_types:
                raise ValueError(
                    "Invalid modality type for feature",
                    feature,
                    ". Choose from:",
                    valid_modality_types,
                )

        if condition == "all":
            continue

        if isinstance(condition, list):
            filtered_models = filtered_models[filtered_models[feature].isin(condition)]
        else:
            filtered_models = filtered_models[filtered_models[feature] == condition]

    if filtered_models.empty:
        warnings.warn("No models match the specified conditions.")

    return filtered_models
