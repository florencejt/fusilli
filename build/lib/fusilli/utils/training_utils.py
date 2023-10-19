"""
Functions for initialising the pytorch lightning logger and trainer, getting final validation metrics
from trained pytorch lightning models, and various functions for setting checkpoint filenames based
on model, parameters, and user-defined strings.
"""

import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from tqdm import tqdm


def get_file_suffix_from_dict(extra_log_string_dict):
    """
    Get the extra name string and tags from the extra_log_string_dict.

    Parameters
    ----------
    extra_log_string_dict : dict
        Extra string to add to the run name. e.g. if you're running
        the same model with different hyperparameters, you can add the hyperparameters.
        Input format {"name": "value"}. In the run name, the extra string will be added
        as "name_value". And a tag will be added as "name_value".

    Returns
    -------
    extra_name_string : str
        Extra name string to add to the some path name.
    extra_tags : list
        List of extra tags to add to the logged run (wandb).
    """

    if extra_log_string_dict is not None:
        extra_name_string = ""
        extra_tags = []
        for key, value in extra_log_string_dict.items():
            extra_name_string += f"_{key}_{str(value)}"
            extra_tags.append(f"{key}_{str(value)}")
    else:
        extra_name_string = ""
        extra_tags = []

    return extra_name_string, extra_tags


def set_logger(params, fold, fusion_model, extra_log_string_dict=None):
    """
    Set the logger for the current run. If params["log"] is True, then the logger is set to
    WandbLogger, otherwise it is set to CSVLogger and the logs are saved to params["loss_log_dir"].

    Parameters
    ----------
    params : dict
        Dictionary of parameters.
    fold : int or None
        Fold number. None if not using kfold.
    fusion_model : class
        Fusion model class.
    extra_log_string_dict : dict
        Extra string to add to the run name. e.g. if you're running
        the same model with different hyperparameters, you can add the hyperparameters.
        Input format {"name": "value"}. In the run name, the extra string will be added
        as "name_value". And a tag will be added as "name_value".
        Default None.

    Returns
    -------
    logger : object
        Pytorch lightning logger object or CSVLogger object if params["log"] is False.
    """

    if hasattr(fusion_model, "__name__"):
        method_name = fusion_model.__name__
    else:
        method_name = fusion_model.__class__.__name__
    modality_type = fusion_model.modality_type
    fusion_type = fusion_model.fusion_type

    extra_name_string, extra_tags = get_file_suffix_from_dict(extra_log_string_dict)

    if params["kfold_flag"]:
        name = f"{method_name}_fold_{fold}{extra_name_string}"
        tags = [modality_type, fusion_type, f"fold_{str(fold)}"] + extra_tags
    else:
        name = f"{method_name}{extra_name_string}"
        tags = [modality_type, fusion_type] + extra_tags

    if params["log"]:
        logger = WandbLogger(
            save_dir=os.getcwd() + "/logs",
            project=params["timestamp"],
            name=name,
            tags=tags,
            log_model=True,
            group=method_name,
            reinit=True,
        )
        logger.experiment.config["method_name"] = method_name
        if extra_log_string_dict is not None:
            for key, value in extra_log_string_dict.items():
                logger.experiment.config[key] = value

    else:  # if params["log"] is False
        logger = CSVLogger(
            save_dir=params["loss_log_dir"],
            name='',
            version=name,
        )

    return logger


def set_checkpoint_name(params, fusion_model, fold=None, extra_log_string_dict=None):
    """
    Set the checkpoint name for the current run of the main fusion model.

    Parameters
    ----------
    params : dict
        Dictionary of parameters.
    fusion_model : class
        Fusion model class.
    fold : int
        Fold number. None if not using kfold.
    extra_log_string_dict : dict
        Extra string to add to the run name. e.g. if you're running
        the same model with different hyperparameters, you can add the hyperparameters.
        Input format {"name": "value"}. In the run name, the extra string will be added
        as "name_value". And a tag will be added as "name_value".
        Default None.

    Returns
    -------
    checkpoint_filename : str
        Checkpoint filename.
    """

    extra_name_string, extra_tags = get_file_suffix_from_dict(extra_log_string_dict)
    if fold is not None:
        checkpoint_filename = (
                fusion_model.__name__
                + "_fold_"
                + str(fold)
                + extra_name_string
                + "_{epoch:02d}"
        )
    else:
        checkpoint_filename = (
                str(fusion_model.__name__) + extra_name_string + "_{epoch:02d}"
        )

    return checkpoint_filename


def get_checkpoint_filenames_for_subspace_models(subspace_method, k=None):
    """
    Get the checkpoint filenames for the subspace models based on the subspace method class
    and the datamodule that is passed into the subspace method class.

    Parameters
    ----------
    subspace_method : class
        Subspace method class.
    k : int
        Fold number. None if not using kfold.
        Default None.

    Returns
    -------
    checkpoint_filenames : list
        List of checkpoint filenames. One for each subspace model in the subspace method class.
    """

    if hasattr(subspace_method.datamodule.fusion_model, "__name__"):
        big_fusion_model_name = subspace_method.datamodule.fusion_model.__name__
    else:
        big_fusion_model_name = (
            subspace_method.datamodule.fusion_model.__class__.__name__
        )

    log_string, _ = get_file_suffix_from_dict(
        subspace_method.datamodule.extra_log_string_dict
    )

    checkpoint_filenames = []
    for subspace_model in subspace_method.subspace_models:
        if k is not None:
            checkpoint_filenames.append(
                "subspace_"
                + big_fusion_model_name
                + "_"
                + subspace_model.__name__
                + "_fold_"
                + str(k)
                + log_string,
            )
        else:
            checkpoint_filenames.append(
                "subspace_"
                + big_fusion_model_name
                + "_"
                + subspace_model.__name__
                + log_string,
            )

    return checkpoint_filenames


def get_checkpoint_filename_for_trained_fusion_model(
        params, model, checkpoint_file_suffix, fold=None
):
    """
    Gets the checkpoint filename for the trained fusion model using the model object.

    Checkpoints should follow the naming convention:

    * fusion_model_name_fold_k_{checkpoint_file_suffix} if fold is not None
    * fusion_model_name_{checkpoint_file_suffix} if fold is None

    Parameters
    ----------
    params : dict
        Dictionary of parameters.
    model : BaseModel
        BaseModel model object instance.
    checkpoint_file_suffix : str
        Checkpoint file suffix.
    fold : int
        Fold number. None if not using kfold.
        Default None.

    Returns
    -------
    checkpoint_filename : str
        Checkpoint filename.
    """
    if checkpoint_file_suffix is None:
        checkpoint_file_suffix = ""

    if fold is None:
        ckpt_path_beginning = model.model.__class__.__name__ + checkpoint_file_suffix
    else:
        ckpt_path_beginning = (
                model.model.__class__.__name__
                + "_fold_"
                + str(fold)
                + checkpoint_file_suffix
        )

    result = [
        filename
        for filename in os.listdir(params["checkpoint_dir"])
        if filename.startswith(ckpt_path_beginning)
    ]

    if len(result) == 0:
        raise ValueError(
            f"Could not find checkpoint file with name {ckpt_path_beginning} in {params['checkpoint_dir']}."
        )
    elif len(result) > 1:
        # if the model is a subspace method, then we need to check if the checkpoint file is for the subspace model
        # or the big fusion model
        # TODO add this check
        raise ValueError(
            f"Found multiple checkpoint files with name {ckpt_path_beginning} in {params['checkpoint_dir']}."
        )
    else:
        checkpoint_filename = result[0]
        checkpoint_filename = os.path.join(
            params["checkpoint_dir"], checkpoint_filename
        )

        return checkpoint_filename


class LitProgressBar(TQDMProgressBar):
    """
    Custom progress bar for pytorch lightning trainer. This is to
    disable the progress bar for validation.

    Parameters
    ----------
    TQDMProgressBar : object
        Pytorch lightning TQDMProgressBar object.
    """

    def init_validation_tqdm(self):
        bar = tqdm(
            disable=True,
        )
        return bar


def init_trainer(
        logger,
        params,
        max_epochs=1000,
        enable_checkpointing=True,
        checkpoint_filename=None,
        own_early_stopping_callback=None,
):
    """
    Initialise the pytorch lightning trainer object.

    Parameters
    ----------
    logger : object
        Pytorch lightning logger object.
    params : dict
        Dictionary of parameters.
    max_epochs : int
        Maximum number of epochs.
        Default 1000.
    enable_checkpointing : bool
        Whether to enable checkpointing. If True, then
        checkpoints will be saved. We use False for the example notebooks in the
        repository/documentation.
        Default True.
    checkpoint_filename : str
        Checkpoint filename.
        Default None if using default checkpointing.
    own_early_stopping_callback : object
        Own early stopping callback object.
        Default None to use default early stopping callback. If you want to use your own early stopping callback,
        then you need to define it in the main training script and pass it here or pass into the datamodule object
        and then it will read it from there.

    Returns
    -------
    trainer : pl.Trainer
        Pytorch lightning trainer object.

    """

    if own_early_stopping_callback is not None:
        early_stop_callback = own_early_stopping_callback
    else:
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=15,
            verbose=False,
            mode="min",
        )

    bar = LitProgressBar()

    callbacks_list = [early_stop_callback, bar]

    if checkpoint_filename is not None:
        checkpoint_callback = ModelCheckpoint(
            filename=checkpoint_filename,
            dirpath=params["checkpoint_dir"],
        )
        callbacks_list.append(checkpoint_callback)

    # check if accelerator and devices are in params given by user
    accelerator = "cpu"  # default
    devices = 1  # default
    if "accelerator" in params.keys():
        accelerator = params["accelerator"]
    if "devices" in params.keys():
        devices = params["devices"]

    trainer = Trainer(
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks_list,
        log_every_n_steps=2,
        # default_root_dir=run_dir,
        logger=logger,
        enable_checkpointing=enable_checkpointing,
    )

    return trainer


def get_final_val_metrics(trainer):
    """
    Get the final validation metrics from the trainer object.

    Parameters
    ----------
    trainer : pl.Trainer
        Pytorch lightning trainer object.

    Returns
    -------
    metric1 : float
        Final validation metric 1.
    metric2 : float
        Final validation metric 2.
    """

    metric_names = trainer.model.metric_names_list

    # raise error if trainer.callback_metrics is empty
    if len(trainer.callback_metrics) == 0:
        raise ValueError("trainer.callback_metrics is empty.")

    # raise error if metric_names[0]_val or metric_names[1]_val is not in trainer.callback_metrics
    if f"{metric_names[0]}_val" not in trainer.callback_metrics.keys():
        raise ValueError(
            f"{metric_names[0]}_val not in trainer.callback_metrics.keys()."
        )
    if f"{metric_names[1]}_val" not in trainer.callback_metrics.keys():
        raise ValueError(
            f"{metric_names[1]}_val not in trainer.callback_metrics.keys()."
        )

    metric1 = trainer.callback_metrics[f"{metric_names[0]}_val"].item()
    metric2 = trainer.callback_metrics[f"{metric_names[1]}_val"].item()

    return metric1, metric2
