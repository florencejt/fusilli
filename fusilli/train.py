"""
Contains the train_and_test function: trains and tests a model and, if k_fold trained, a fold.
"""

from fusilli.fusionmodels.base_model import BaseModel
from fusilli.utils.training_utils import (
    get_final_val_metrics,
    init_trainer,
    set_logger,
    set_checkpoint_name,
)
import wandb
from fusilli.utils import model_modifier
from lightning.pytorch.loggers import CSVLogger
from fusilli.utils.csv_loss_plotter import plot_loss_curve


def train_and_test(
        data_module,
        k,
        fusion_model,
        kfold,
        extra_log_string_dict=None,
        layer_mods=None,
        max_epochs=1000,
        enable_checkpointing=True,
        show_loss_plot=False,
        wandb_logging=False,
        project_name=None,
        training_modifications=None,
        metrics_list=None,
):
    """
    Trains and tests a model and, if k_fold trained, a fold.

    Parameters
    ----------
    data_module : pytorch lightning data module
        Data module.
        Contains the train and val dataloaders.
    k : int
        Fold number.
    fusion_model : class
        Fusion model class.
    kfold : bool
        Whether to train a kfold model.
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
    show_loss_plot : bool
        Whether to show the loss plot. Default False.
        If True, the loss plot will be shown after training with ``plt.show()``
        If False, the loss plot will be saved to the log directory.
    wandb_logging : bool
        Whether to log to Weights and Biases. Default False.
    project_name : str or None
        Name of the project to log to in Weights and Biases. Default None.
        If None, the project name will be called "fusilli".
    training_modifications : dict
        Dictionary of training modifications. Used to modify the training process. Keys could be "accelerator", "devices"
    metrics_list : list
        List of metrics to use for model evaluation. Default None.
        If None, the metrics will be automatically selected based on the prediction task
        (AUROC, accuracy for binary/multiclass, R2 and MAE for regression).
        The first metric in the list will be used in the comparison evaluation figures to rank the models' performances.
        Length must be 2 or more.

    Returns
    -------
    model : pytorch lightning model
        Trained model.
    trainer : pytorch lightning trainer
        Trained trainer.
    metric_1 : float
        Metric 1 (depends on metric_name_list and prediction_task.
    metric_2 : float
        Metric 2 (depends on metric_name_list and prediction_task.
    val_reals : list
        List of validation real values.
    val_preds : list
        List of validation predicted values.
    """

    # define checkpoint filename
    if kfold:
        if enable_checkpointing:
            checkpoint_filename = set_checkpoint_name(
                fusion_model,
                fold=k,
                extra_log_string_dict=extra_log_string_dict,
            )
        else:
            checkpoint_filename = None

        if fusion_model.fusion_type == "graph":
            data_module = data_module[k]
            output_paths = data_module.output_paths
            train_dataloader = data_module.train_dataloader()
            val_dataloader = data_module.val_dataloader()
        else:
            output_paths = data_module.output_paths
            train_dataloader = data_module.train_dataloader(fold_idx=k)
            val_dataloader = data_module.val_dataloader(fold_idx=k)

    else:
        if enable_checkpointing:
            checkpoint_filename = set_checkpoint_name(
                fusion_model=fusion_model,
                extra_log_string_dict=extra_log_string_dict,
            )
        else:
            checkpoint_filename = None

        train_dataloader = data_module.train_dataloader()
        val_dataloader = data_module.val_dataloader()
        output_paths = data_module.output_paths

    logger = set_logger(fold=k,
                        project_name=project_name,
                        output_paths=output_paths,
                        fusion_model=fusion_model,
                        extra_log_string_dict=extra_log_string_dict,
                        wandb_logging=wandb_logging,
                        )  # set logger

    trainer = init_trainer(
        logger,
        output_paths=output_paths,
        max_epochs=max_epochs,
        enable_checkpointing=enable_checkpointing,
        checkpoint_filename=checkpoint_filename,
        own_early_stopping_callback=data_module.own_early_stopping_callback,
        training_modifications=training_modifications,
    )  # init trainer

    # initialise model with pytorch lightning framework, hence pl_model
    pl_model = BaseModel(
        fusion_model(
            prediction_task=data_module.prediction_task,
            data_dims=data_module.data_dims,  # data_dims is a list of tuples
            multiclass_dimensions=data_module.multiclass_dimensions,
        ),
        metrics_list=metrics_list,
    )

    # modify model architecture if layer_mods is not None
    if layer_mods is not None:
        pl_model.model = model_modifier.modify_model_architecture(
            pl_model.model, layer_mods
        )

    # graph-methods use masks to select train and val nodes rather than train and val dataloaders
    if pl_model.model.fusion_type == "graph":
        pl_model.train_mask = data_module.input_train_nodes
        pl_model.val_mask = data_module.input_test_nodes

    # fit model
    trainer.fit(
        pl_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # test model
    trainer.validate(pl_model, val_dataloader)

    # get final validation metrics
    final_val_metrics = get_final_val_metrics(trainer)
    pl_model.final_val_metrics = final_val_metrics

    # if logger is CSVLogger, plot loss curve
    if isinstance(logger, CSVLogger):
        plot_loss_curve(figures_path=output_paths["figures"], logger=logger, show=show_loss_plot)

    return pl_model


def _store_trained_model(trained_model, trained_models_dict):
    """
    Stores the trained model to a dictionary.
    If model type is already in dictionary (e.g. if it's a kfold model), append the model to
    the list of models.

    Parameters
    ----------
    trained_model : pytorch lightning model
        Trained model.
    trained_models_dict : dict
        Dictionary of trained models.

    Returns
    -------
    trained_models_dict : dict
        Dictionary of trained models.
    """

    # get model name
    classname = trained_model.model.__class__.__name__

    # if model is already in dictionary, we're training a kfold model
    if classname in trained_models_dict:
        # if the model is already a list, append the new model to the list
        # this is for when we're training a kfold model and on the third fold onwards
        if isinstance(trained_models_dict[classname], list):
            trained_models_dict[classname].append(trained_model)
        # if the model is not a list, make it a list and append the new model to the list
        # this is for when we're training a kfold model and on the second fold
        else:
            trained_models_dict[classname] = [
                trained_models_dict[classname],
                trained_model,
            ]
    else:
        # If the model is not in the dictionary, add it as a new key-value pair
        # This is for when we're training a single model with train/test split or
        # when we're training a kfold model and on the first fold
        trained_models_dict[classname] = [trained_model]

    return trained_models_dict


def train_and_save_models(
        data_module,
        fusion_model,
        wandb_logging=False,
        extra_log_string_dict=None,
        layer_mods=None,
        max_epochs=1000,
        enable_checkpointing=True,
        show_loss_plot=False,
        project_name=None,
        metrics_list=None,
):
    """
    Trains/tests the model and saves the trained model to a dictionary for further analysis.
    If the training type is kfold, it will train and test the model for each fold and store the
    trained models in a list.

    Parameters
    ----------
    data_module : pytorch lightning data module
        Data module.
    fusion_model : class
        Fusion model class.
    wandb_logging : bool
        Whether to log to wandb. Default False.
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
    show_loss_plot : bool
        Whether to show the loss plot. Default False.
    project_name : str or None
        Name of the project to log to in Weights and Biases. Default None.
        If None, the project name will be called "fusilli".
    metrics_list : list
        List of metrics to use for model evaluation. Default None.
        If None, the metrics will be automatically selected based on the prediction task
        (AUROC, accuracy for binary/multiclass, R2 and MAE for regression).
        The first metric in the list will be used in the comparison evaluation figures to rank the models' performances.
        Length must be 2 or more.

    Returns
    -------
    trained_models_list : list
        List of trained models.
        Length of list is 1 if training type is train/test split.
        Length of list is num_k if training type is kfold.
    """

    # trained_models_dict = {}
    trained_models_list = []

    # checking to see if our model is a kfold model

    if hasattr(data_module, "num_folds") and data_module.num_folds is not None:
        kfold = True
        num_folds = data_module.num_folds
    elif isinstance(data_module, list):
        if hasattr(data_module[0], "num_folds") and data_module[0].num_folds is not None:
            kfold = True
            num_folds = data_module[0].num_folds
    else:
        kfold = False
        num_folds = None

    if kfold:
        for k in range(num_folds):
            trained_model = train_and_test(
                data_module=data_module,
                k=k,
                fusion_model=fusion_model,
                kfold=kfold,
                extra_log_string_dict=extra_log_string_dict,
                layer_mods=layer_mods,
                max_epochs=max_epochs,
                enable_checkpointing=enable_checkpointing,
                show_loss_plot=show_loss_plot,
                wandb_logging=wandb_logging,
                project_name=project_name,
                metrics_list=metrics_list,
            )

            trained_models_list.append(trained_model)

            if wandb_logging:
                wandb.finish()

    else:
        trained_model = train_and_test(
            data_module=data_module,
            k=None,
            fusion_model=fusion_model,
            kfold=kfold,
            extra_log_string_dict=extra_log_string_dict,
            layer_mods=layer_mods,
            max_epochs=max_epochs,
            enable_checkpointing=enable_checkpointing,
            show_loss_plot=show_loss_plot,
            wandb_logging=wandb_logging,
            project_name=project_name,
            metrics_list=metrics_list,
        )

        trained_models_list.append(trained_model)

        if wandb_logging:
            wandb.finish()

    return trained_models_list
