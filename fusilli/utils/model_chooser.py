"""
This module contains the function to filter the fusion models based on the conditions specified by the user.
Conditions are specified in a dictionary, where the keys are the features to filter by and the values are the
conditions to filter by. The function returns a dataframe containing the filtered models.
"""

import importlib
import pandas as pd
import warnings

# list of dictionaries containing the fusion models' names and paths
# this must be updated whenever a new fusion model is added
fusion_model_dict = [
    {"name": "Tabular1Unimodal", "path": "fusionmodels.unimodal.tabular1"},
    {"name": "Tabular2Unimodal", "path": "fusionmodels.unimodal.tabular2"},
    {"name": "ImgUnimodal", "path": "fusionmodels.unimodal.image"},
    {
        "name": "ConcatTabularFeatureMaps",
        "path": "fusionmodels.tabularfusion.concat_feature_maps",
    },
    {
        "name": "ConcatImageMapsTabularData",
        "path": "fusionmodels.tabularimagefusion.concat_img_maps_tabular_data",
    },
    {
        "name": "ConcatTabularData",
        "path": "fusionmodels.tabularfusion.concat_data",
    },
    {
        "name": "ConcatImageMapsTabularMaps",
        "path": "fusionmodels.tabularimagefusion.concat_img_maps_tabular_maps",
    },
    {
        "name": "TabularChannelWiseMultiAttention",
        "path": "fusionmodels.tabularfusion.channelwise_att",
    },
    {
        "name": "ImageChannelWiseMultiAttention",
        "path": "fusionmodels.tabularimagefusion.channelwise_att",
    },
    {
        "name": "CrossmodalMultiheadAttention",
        "path": "fusionmodels.tabularimagefusion.crossmodal_att",
    },
    {
        "name": "TabularCrossmodalMultiheadAttention",
        "path": "fusionmodels.tabularfusion.crossmodal_att",
    },
    {
        "name": "TabularDecision",
        "path": "fusionmodels.tabularfusion.decision",
    },
    {
        "name": "ImageDecision",
        "path": "fusionmodels.tabularimagefusion.decision",
    },
    {"name": "MCVAE_tab", "path": "fusionmodels.tabularfusion.mcvae_model"},
    {
        "name": "ConcatImgLatentTabDoubleTrain",
        "path": "fusionmodels.tabularimagefusion.concat_img_latent_tab_doubletrain",
    },
    {
        "name": "ConcatImgLatentTabDoubleLoss",
        "path": "fusionmodels.tabularimagefusion.concat_img_latent_tab_doubleloss",
    },
    {"name": "EdgeCorrGNN", "path": "fusionmodels.tabularfusion.edge_corr_gnn"},
    {
        "name": "DAETabImgMaps",
        "path": "fusionmodels.tabularimagefusion.denoise_tab_img_maps",
    },
    {
        "name": "AttentionWeightedGNN",
        "path": "fusionmodels.tabularfusion.attention_weighted_GNN",
    },
    {
        "name": "AttentionAndSelfActivation",
        "path": "fusionmodels.tabularfusion.attention_and_activation",
    },
    {
        "name": "ActivationFusion",
        "path": "fusionmodels.tabularfusion.activation",
    }
]


def all_model_importer(fusion_model_dict, skip_models=None):
    """
    Imports all the fusion models in the fusion_model_dict.

    Parameters
    ----------
    fusion_model_dict : list
        List of dictionaries containing all the fusion models' names and paths.
        Names mean the name of the class, and paths mean the path to the .py file
        containing the class.
        Note: this must be updated whenever a new fusion model is added.
    skip_models : list
        List of models to skip when importing. Default is None.
        The list should consist of the class names of the models to skip e.g. ["TabularDecision", "ImgUnimodal"]. You might skip models if some are not working properly for you.

    Returns
    -------
    fusion_models : list
        List of all the fusion models class objects
    fusion_model_dict_copy : list
        List of dictionaries containing all the fusion models' names and paths, without the models that were skipped.
    """

    # create copy of fusion_model_dict so we can remove skip_models from it
    fusion_model_dict_copy = fusion_model_dict.copy()

    fusion_models = []
    for model in fusion_model_dict:
        module_name = model["name"]

        # if we're skipping importing some models
        if skip_models is not None:
            if module_name in skip_models:
                # remove model from fusion_model_dict_copy
                fusion_model_dict_copy.remove(model)
                continue

        module_path = "fusilli." + model["path"]

        module = importlib.import_module(module_path)
        module_class = getattr(module, module_name)

        fusion_models.append(module_class)

    return fusion_models, fusion_model_dict_copy


def get_models(conditions_dict, skip_models=None, fusion_model_dict=fusion_model_dict, ):
    """Filters the models based on the conditions specified by the user.

    Parameters
    ----------
    conditions_dict : dict
        Dictionary containing the conditions to filter the models.
        Structure: {feature1: condition, feature2: condition, ...}
        or {feature1: [condition1, condition2, ...], feature2: [condition1, ...], ...}

        Accepted features and accepted conditions:

        - "fusion_type": "unimodal", "operation", "attention", "subspace", "graph", or "all"
        - "modality_type": "tabular1", "tabular2", "img", "tabular_tabular", "tabular_image", or "all"
        - "method_name": any method name currently implemented (e.g. "Tabular decision"), or "all"
        - "class_name": any model name currently implemented (e.g. "TabularDecision"), or "all"

        Example: To get all the models that are uni-modal and attention-based, the dictionary would be:

        .. code-block:: python

            conditions_dict = {
                "fusion_type": ["unimodal", "operation"],
                "modality_type": "all",
                }

    fusion_model_dict : list
        List of dictionaries containing the fusion models' names and paths. Default is fusion_model_dict.

    skip_models : list
        List of models to skip when importing. Default is None.
        The list should consist of the class names of the models to skip e.g. ["TabularDecision", "ImgUnimodal"].  You might skip models if some are not working properly for you.


    Returns
    -------
    filtered_models : pd.DataFrame
        Dataframe containing the filtered models.

        Column names:

        - "method_name": name of the model (e.g. "Tabular decision")
        - "fusion_type": type of fusion (e.g. "operation")
        - "modality_type": type of modality (e.g. "tabular_tabular")
        - "class_name": name of the class (e.g. "TabularDecision")
        - "method_path": path to the method's py file (e.g. "fusilli.fusionmodels.tabular_decision")

    """

    # raise error if condition is not "all" and feature is not one of the options
    valid_features = ["fusion_type", "modality_type", "method_name", "class_name"]
    valid_fusion_types = [
        "unimodal",
        "operation",
        "attention",
        "subspace",
        "graph",
        "tensor",
    ]
    valid_modality_types = ["tabular1", "tabular2", "img", "tabular_tabular", "tabular_image"]

    fusion_models, fusion_model_dict_without_skips = all_model_importer(fusion_model_dict, skip_models=skip_models)

    method_names = [
        fusion_models[i].method_name for i, model in enumerate(fusion_model_dict_without_skips)
    ]
    fusion_types = [
        fusion_models[i].fusion_type for i, model in enumerate(fusion_model_dict_without_skips)
    ]
    modality_types = [
        fusion_models[i].modality_type for i, model in enumerate(fusion_model_dict_without_skips)
    ]

    class_names = [fusion_models[i].__name__ for i, model in enumerate(fusion_models)]

    method_paths = [
        "fusilli." + model["path"] for i, model in enumerate(fusion_model_dict_without_skips)
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
                        "Invalid fusion type for feature",
                        feature,
                        ":",
                        invalid_fusion_types,
                        ". Choose from:",
                        valid_fusion_types,
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
                        "Invalid modality type for feature",
                        feature,
                        ":",
                        invalid_modality_types,
                        ". Choose from:",
                        valid_modality_types,
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


def import_chosen_fusion_models(model_conditions, skip_models=None):
    """
    Imports the fusion models specified by the user.

    Parameters
    ----------
    model_conditions : dict
        Dictionary containing the conditions to filter the models.
        Structure: {feature1: condition, feature2: condition, ...}
        or {feature1: [condition1, condition2, ...], feature2: [condition1, ...], ...}

        Accepted features and accepted conditions:

        - "fusion_type": "unimodal", "operation", "attention", "subspace", "graph", or "all"
        - "modality_type": "tabular1", "tabular2", "img", "tabular_tabular", "tabular_image", or "all"
        - "method_name": any method name currently implemented (e.g. "Tabular decision"), or "all"
        - "class_name": any model name currently implemented (e.g. "TabularDecision"), or "all"

        Example: To get all the models that are uni-modal and attention-based, the dictionary would be:

        .. code-block:: python

            conditions_dict = {
                "fusion_type": ["unimodal", "operation"],
                "modality_type": "all",
                }

    skip_models : list
        List of models to skip when importing. Default is None.
        The list should consist of the class names of the models to skip e.g. ["TabularDecision", "ImgUnimodal"]. You might skip models if some are not working properly for you.

    Returns
    -------
    fusion_models : list
        List of all the fusion models class objects
    """
    imported_models = get_models(model_conditions, skip_models)
    print("Imported methods:")
    print(imported_models.method_name.values)

    fusion_models = []
    for index, row in imported_models.iterrows():
        module = importlib.import_module(row["method_path"])
        module_class = getattr(module, row["class_name"])
        fusion_models.append(module_class)

    return fusion_models
