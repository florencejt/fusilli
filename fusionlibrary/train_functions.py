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
    changed = False
    for model_name in architecture_modification:
        # ~~~~~~ modifying all fusion models ~~~~~~
        if model_name == "all":
            for i, layer_group in enumerate(architecture_modification[model_name]):
                if hasattr(model, layer_group):
                    setattr(
                        model,
                        layer_group,
                        architecture_modification[model_name][layer_group],
                    )
                    # RESET FUSED LAYERS
                    if (layer_group != "fused_layers") and (
                        i == len(architecture_modification[model_name]) - 1
                    ):  # don't reset fused layers if we're modifying fused layers anyway
                        if hasattr(model, "calc_fused_layers"):
                            model.calc_fused_layers()  # reset fused layers with new fused_dim from new architecture
                    changed = True

        # ~~~~~~ modifying specific fusion models ~~~~~~
        if model_name == model.__class__.__name__:
            for i, layer_group in enumerate(architecture_modification[model_name]):
                # if we need to access a layer within a layer group
                split_layer_group = layer_group.split(".")

                if len(split_layer_group) > 1:
                    get_attr_1 = getattr(model, split_layer_group[0])
                    for layer_layer in split_layer_group[1:-1]:
                        get_attr_1 = getattr(get_attr_1, layer_layer)
                else:
                    get_attr_1 = model

                if hasattr(get_attr_1, split_layer_group[-1]):
                    setattr(
                        get_attr_1,
                        split_layer_group[-1],
                        architecture_modification[model_name][layer_group],
                    )
                    changed = True

                # RESET FUSED LAYERS
                if (layer_group != "fused_layers") and (
                    i == len(architecture_modification[model_name]) - 1
                ):  # don't reset fused layers if we're modifying fused layers anyway
                    print("resetting fused layers", layer_group, "on", get_attr_1)
                    if hasattr(get_attr_1, "calc_fused_layers"):
                        get_attr_1.calc_fused_layers()  # reset fused layers with new fused_dim from new architecture

    # if not changed:
    #     raise ValueError("NOTHING CHANGED IN THIS MODEL FOR TESTING")

    return model


def train_and_test(
    dm,
    params,
    k,
    fusion_model,
    init_model,
    extra_log_string_dict=None,
    layer_mods=None,
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
    init_model : class
        Init model class (initialized empty for method details).
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

    logger = set_logger(params, k, init_model, extra_log_string_dict)  # set logger

    trainer = init_trainer(logger, max_epochs=3)  # init trainer

    model = BaseModel(
        fusion_model(
            pred_type=params[
                "pred_type"
            ],  # pred_type is a string (binary, regression, multiclass)
            data_dims=dm.data_dims,  # data_dims is a list of tuples
            params=params,  # params is a dict
        )
    )

    print("dm.data_dims", dm.data_dims)

    # TODO add in bit to let user choose structure of model
    if layer_mods is not None:
        model.model = modify_model_architecture(model.model, layer_mods)

    # if architecture_modification is not None:
    #    for layer_group in architecture_modification:
    #        if layer_group exists in model:
    #            modify layer_group
    #       else:
    #            print(f"Layer group {layer_group} does not exist in model")

    # graph-methods use masks to select train and val nodes rather than train and val dataloaders
    # train and val dataloaders are still used for graph but they're identical
    if model.fusion_type == "graph":
        model.train_mask = dm.input_train_nodes
        model.val_mask = dm.input_test_nodes

    # fit model
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # test model
    trainer.validate(model, val_dataloader)

    metric_1, metric_2 = get_final_val_metrics(trainer)

    model.metric1 = metric_1
    model.metric2 = metric_2

    return model


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
    trained_models_dict,
    data_module,
    params,
    fusion_model,
    init_model,
    extra_log_string_dict=None,
    layer_mods=None,
):
    """
    Trains/tests the model and saves the trained model to a dictionary for further analysis.
    If the training type is kfold, it will train and test the model for each fold and store the
    trained models in a list under the model type key.

    Parameters
    ----------
    trained_models_dicts : dict
        Dictionary of trained models.
    data_module : pytorch lightning data module
        Data module.
    params : dict
        Dictionary of parameters.
    fusion_model : class
        Fusion model class.
    init_model : class
        Init model class (initialized empty for method details).
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

    """
    if params["kfold_flag"]:
        for k in range(params["num_k"]):
            trained_model = train_and_test(
                dm=data_module,
                params=params,
                k=k,
                fusion_model=fusion_model,
                init_model=init_model,
                extra_log_string_dict=extra_log_string_dict,
                layer_mods=layer_mods,
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
            init_model=init_model,
            extra_log_string_dict=extra_log_string_dict,
            layer_mods=layer_mods,
        )
        trained_models_dict = store_trained_model(trained_model, trained_models_dict)

        if params["log"]:
            wandb.finish()

    return trained_models_dict
