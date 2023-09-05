"""
Contains the train_and_test function: trains and tests a model and, if k_fold trained, a fold.
"""

from fusionlibrary.fusion_models.base_pl_model import BaseModel
from fusionlibrary.utils.pl_utils import (
    get_final_val_metrics,
    init_trainer,
    set_logger,
)
import wandb


def modify_model_architecture(model, architecture_modification):
    """
    Modify the architecture of a deep learning model based on the provided configuration.

    Args:
        model (nn.Module): The original deep learning model.
        architecture_modification (dict): A dictionary containing architecture modifications.

    Returns:
        nn.Module: The modified deep learning model.
    """
    for model_name, layer_groups in architecture_modification.items():
        # Modify layers for all specified models
        if model_name == "all":
            for layer_group, modification in layer_groups.items():
                if hasattr(model, layer_group):
                    setattr(model, layer_group, modification)
                    # if layer_group != "fused_layers":
                    #     # if we;re on the last modification
                    #     if layer_group == list(layer_groups.keys())[-1]:
                    print("Changed", layer_group, "in", model_name)
            reset_fused_layers(model)

        # Modify layers for a specific model class
        elif model_name == model.__class__.__name__:
            for layer_group, modification in layer_groups.items():
                nested_attr = get_nested_attr(model, layer_group)

                if hasattr(nested_attr, layer_group.split(".")[-1]):
                    setattr(nested_attr, layer_group.split(".")[-1], modification)
                    print("Changed", layer_group.split(".")[-1], "in", model_name)
                    # if layer_group != "fused_layers":
                    #     # if we;re on the last modification
                    #     if layer_group == list(layer_groups.keys())[-1]:
                reset_fused_layers(nested_attr)

    return model


def get_nested_attr(obj, attr_path):
    """
    Get a nested attribute from an object using dot-separated path.

    Args:
        obj (object): The object to retrieve the nested attribute from.
        attr_path (str): Dot-separated path to the nested attribute.

    Returns:
        object: The nested attribute if found, otherwise None.
    """
    attributes = attr_path.split(".")

    if len(attributes) > 1:  # if we're looking for a nested attribute
        attr = getattr(obj, attributes[0])
        for i in range(1, len(attributes)):
            attr = getattr(attr, attributes[i])
    else:
        attr = obj

    return attr


def reset_fused_layers(obj):
    """
    Reset fused layers of a model if the reset method is available.

    Args:
        obj (nn.Module): The model to reset fused layers for.
    """
    if hasattr(obj, "calc_fused_layers"):
        obj.calc_fused_layers()


def train_and_test(
    dm,
    params,
    k,
    fusion_model,
    extra_log_string_dict=None,
    layer_mods=None,
    max_epochs=1000,
):
    """
    Trains and tests a model and, if k_fold trained, a fold.

    Parameters
    ----------
    dm : pytorch lightning data module
        Data module.
    params : dict
        Dictionary of parameters.
    k : int
        Fold number.
    fusion_model : class
        Fusion model class.
    metric_name_list : list
        List of metric names.
    method_name : str
        Name of the method.
    extra_log_string_dict : dict
        Dictionary of extra log strings. Extra string to add to the run name during logging.
            e.g. if you're running the same model with different hyperparameters, you can add
            the hyperparameters.
            Input format {"name": "value"}. In the run name, the extra string will be added
            as "name_value". And a tag will be added as "name_value".
    layer_mods : dict
        Dictionary of layer modifications. Used to modify the architecture of the model.
        Input format {"model": {"layer_group": "modification"}, ...}.
        e.g. {"TabularCrossmodalAttention": {"mod1_layers": new mod 1 layers nn.ModuleDict}}
        Default None.
    max_epochs : int
        Maximum number of epochs. Default 1000.

    Returns
    -------
    model : pytorch lightning model
        Trained model.
    trainer : pytorch lightning trainer
        Trained trainer.
    metric_1 : float
        Metric 1 (depends on metric_name_list and params["pred_type"]).
    metric_2 : float
        Metric 2 (depends on metric_name_list and params["pred_type"]).
    val_reals : list
        List of validation real values.
    val_preds : list
        List of validation predicted values.
    """

    if params["kfold_flag"]:
        if fusion_model.fusion_type == "graph":
            # graph k-fold loader is different to normal k-fold loader
            dm = dm[k]
            train_dataloader = dm.train_dataloader()
            val_dataloader = dm.val_dataloader()
        else:
            train_dataloader = dm.train_dataloader(fold_idx=k)
            val_dataloader = dm.val_dataloader(fold_idx=k)

        # update plot file name suffix
        # plot_file_suffix = f"_{method_name}_rep_{rep_n}_fold_{k}"
    else:
        train_dataloader = dm.train_dataloader()
        val_dataloader = dm.val_dataloader()
        # plot_file_suffix = f"_{method_name}_rep_{rep_n}"

    logger = set_logger(params, k, fusion_model, extra_log_string_dict)  # set logger

    trainer = init_trainer(logger, max_epochs=max_epochs)  # init trainer

    # initialise model with pytorch lightning framework, hence pl_model
    pl_model = BaseModel(
        fusion_model(
            pred_type=params[
                "pred_type"
            ],  # pred_type is a string (binary, regression, multiclass)
            data_dims=dm.data_dims,  # data_dims is a list of tuples
            params=params,  # params is a dict
        )
    )

    if layer_mods is not None:
        pl_model.model = modify_model_architecture(pl_model.model, layer_mods)

    # graph-methods use masks to select train and val nodes rather than train and val dataloaders
    # train and val dataloaders are still used for graph but they're identical
    if pl_model.model.fusion_type == "graph":
        pl_model.train_mask = dm.input_train_nodes
        pl_model.val_mask = dm.input_test_nodes

    # fit model
    trainer.fit(
        pl_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # test model
    trainer.validate(pl_model, val_dataloader)

    metric_1, metric_2 = get_final_val_metrics(trainer)

    pl_model.metric1 = metric_1
    pl_model.metric2 = metric_2

    return pl_model


def store_trained_model(trained_model, trained_models_dict):
    """
    Stores the trained model to a dictionary.
    If model type is already in dictionary (e.g. if it's a kfold model), append the model to
    the list of models.
    """

    classname = trained_model.model.__class__.__name__

    if classname in trained_models_dict:
        # If the model is already in the dictionary, convert the existing value to a list
        if isinstance(trained_models_dict[classname], list):
            trained_models_dict[classname].append(trained_model)
        else:
            trained_models_dict[classname] = [
                trained_models_dict[classname],
                trained_model,
            ]
    else:
        # If the model is not in the dictionary, add it as a new key-value pair
        trained_models_dict[classname] = trained_model

    return trained_models_dict


def train_and_save_models(
    data_module,
    params,
    fusion_model,
    extra_log_string_dict=None,
    layer_mods=None,
    max_epochs=1000,
):
    """
    Trains/tests the model and saves the trained model to a dictionary for further analysis.
    If the training type is kfold, it will train and test the model for each fold and store the
    trained models in a list under the model type key.

    Parameters
    ----------
    data_module : pytorch lightning data module
        Data module.
    params : dict
        Dictionary of parameters.
    fusion_model : class
        Fusion model class.
    extra_log_string_dict : dict
        Dictionary of extra log strings. Extra string to add to the run name during logging.
        e.g. if you're running the same model with different hyperparameters, you can add
        the hyperparameters.
        Input format {"name": "value"}. In the run name, the extra string will be added
        as "name_value". And a tag will be added as "name_value".
    layer_mods : dict
        Dictionary of layer modifications. Used to modify the architecture of the model.
        Input format {"model": {"layer_group": "modification"}, ...}.
        e.g. {"TabularCrossmodalAttention": {"mod1_layers": new mod 1 layers nn.ModuleDict}}
        Default None.
    max_epochs : int
        Maximum number of epochs. Default 1000.

    Returns
    -------
    trained_models_dict : dict
        Dictionary of trained model (key is model name, value is either single trained model for
        train/test training or a list of trained models for k-fold training).

    """

    trained_models_dict = {}

    if params["kfold_flag"]:
        for k in range(params["num_k"]):
            trained_model = train_and_test(
                dm=data_module,
                params=params,
                k=k,
                fusion_model=fusion_model,
                extra_log_string_dict=extra_log_string_dict,
                layer_mods=layer_mods,
                max_epochs=max_epochs,
            )
            trained_models_dict = store_trained_model(
                trained_model, trained_models_dict
            )

            if params["log"]:
                wandb.finish()

    else:
        trained_model = train_and_test(
            dm=data_module,
            params=params,
            k=None,
            fusion_model=fusion_model,
            extra_log_string_dict=extra_log_string_dict,
            layer_mods=layer_mods,
            max_epochs=max_epochs,
        )
        trained_models_dict = store_trained_model(trained_model, trained_models_dict)

        if params["log"]:
            wandb.finish()

    return trained_models_dict
