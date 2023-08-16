"""
Contains the train_and_test function: trains and tests a model for a given
repetition and, if k_fold trained, a fold.
"""

from fusionlibrary.fusion_models.base_pl_model import BaseModel
from fusionlibrary.utils.pl_utils import (
    get_final_val_metrics,
    init_trainer,
    set_logger,
)


def train_and_test(
    dm,
    params,
    rep_n,
    k,
    fusion_model,
    init_model,
    metric_name_list,
    method_name,
):
    """
    Trains and tests a model for a given repetition and, if k_fold trained, a fold.

    Parameters
    ----------
    dm : pytorch lightning data module
        Data module.
    params : dict
        Dictionary of parameters.
    rep_n : int
        Repetition number.
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
        if init_model.fusion_type == "graph":
            # graph k-fold loader is different to normal k-fold loader
            dm = dm[k]
            train_dataloader = dm.train_dataloader()
            val_dataloader = dm.val_dataloader()
        else:
            train_dataloader = dm.train_dataloader(fold_idx=k)
            val_dataloader = dm.val_dataloader(fold_idx=k)

        # update plot file name suffix
        plot_file_suffix = f"_{method_name}_rep_{rep_n}_fold_{k}"
    else:
        train_dataloader = dm.train_dataloader()
        val_dataloader = dm.val_dataloader()
        plot_file_suffix = f"_{method_name}_rep_{rep_n}"

    logger = set_logger(params, rep_n, k, init_model)  # set logger

    trainer = init_trainer(logger)  # init trainer

    model = BaseModel(
        fusion_model(
            pred_type=params[
                "pred_type"
            ],  # pred_type is a string (binary, regression, multiclass)
            data_dims=dm.data_dims,  # data_dims is a list of tuples
            params=params,  # params is a dict
        )
    )

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

    # update results
    metric_1, metric_2 = get_final_val_metrics(metric_name_list, trainer)

    # plot end of run figures
    eval_figs, val_reals, val_preds = model.plot_eval_figs(
        train_dataloader,
        val_dataloader,
        plot_file_suffix,
    )

    return model, trainer, metric_1, metric_2, val_reals, val_preds
