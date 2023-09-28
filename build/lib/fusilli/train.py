"""
Contains the train_and_test function: trains and tests a model and, if k_fold trained, a fold.
"""

from fusilli.fusion_models.base_model import BaseModel
from fusilli.utils.training_utils import (
    get_final_val_metrics,
    init_trainer,
    set_logger,
)
import wandb
import warnings
import inspect
from fusilli.utils import model_modifier


def train_and_test(
    dm,
    params,
    k,
    fusion_model,
    extra_log_string_dict=None,
    layer_mods=None,
    max_epochs=1000,
    enable_checkpointing=True,
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
    enable_checkpointing : bool
        Whether to enable checkpointing. Default True.

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

    else:
        train_dataloader = dm.train_dataloader()
        val_dataloader = dm.val_dataloader()

    logger = set_logger(params, k, fusion_model, extra_log_string_dict)  # set logger

    trainer = init_trainer(
        logger, max_epochs=max_epochs, enable_checkpointing=enable_checkpointing
    )  # init trainer

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
        pl_model.model = model_modifier.modify_model_architecture(
            pl_model.model, layer_mods
        )

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


def store_trained_model(trained_model, best_checkpoint_pth, trained_models_dict):
    """
    Stores the trained model to a dictionary.
    If model type is already in dictionary (e.g. if it's a kfold model), append the model to
    the list of models.
    """

    classname = trained_model.model.__class__.__name__

    # print(trained_models_dict)
    # key 1: trained_model, values: list of trained models
    # key 2: checkpoint_path, value: list of best checkpoint path
    # key 3: subspace_model, value: list of trained subspace models
    # key 4: subspace_checkpoint_path, value: list of best checkpoint path for subspace model

    if classname in trained_models_dict:
        # If the model is already in the dictionary, convert the existing value to a list
        if isinstance(trained_models_dict[classname], list):
            trained_models_dict[classname].append([trained_model, best_checkpoint_pth])
        else:
            trained_models_dict[classname] = [
                trained_models_dict[classname],
                [trained_model, best_checkpoint_pth],
            ]
    else:
        # If the model is not in the dictionary, add it as a new key-value pair
        trained_models_dict[classname] = [[trained_model, best_checkpoint_pth]]

    return trained_models_dict


def train_and_save_models(
    data_module,
    params,
    fusion_model,
    extra_log_string_dict=None,
    layer_mods=None,
    max_epochs=1000,
    enable_checkpointing=True,
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
    enable_checkpointing : bool
        Whether to enable checkpointing. Default True.

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
                enable_checkpointing=enable_checkpointing,
            )

            # print("Trainer", trained_model.trainer)

            trained_models_dict = store_trained_model(
                trained_model,
                trained_model.trainer.checkpoint_callback.best_model_path,
                trained_models_dict,
            )
            # print(trained_model.trainer.checkpoint_callback.best_model_path)

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
            enable_checkpointing=enable_checkpointing,
        )

        # print(
        #     "Trained model:",
        #     trained_model.state_dict()["model.final_prediction.0.weight"],
        # )
        trained_models_dict = store_trained_model(
            trained_model,
            trained_model.trainer.checkpoint_callback.best_model_path,
            trained_models_dict,
        )

        if params["log"]:
            wandb.finish()

    return trained_models_dict
