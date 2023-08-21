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
    k,
    fusion_model,
    init_model,
    extra_log_string_dict=None,
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
    extra_log_string_dict : dict
        Dictionary of extra log strings. Extra string to add to the run name during logging.
            e.g. if you're running the same model with different hyperparameters, you can add
            the hyperparameters.
            Input format {"name": "value"}. In the run name, the extra string will be added
            as "name_value". And a tag will be added as "name_value".

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

    # if params["kfold_flag"]:
    #     if init_model.fusion_type == "graph":
    #         # graph k-fold loader is different to normal k-fold loader
    #         dm = dm[k]
    #         train_dataloader = dm.train_dataloader()
    #         val_dataloader = dm.val_dataloader()
    #     else:
    #         train_dataloader = dm.train_dataloader(fold_idx=k)
    #         val_dataloader = dm.val_dataloader(fold_idx=k)

    #     # update plot file name suffix
    #     plot_file_suffix = f"_{method_name}_rep_{rep_n}_fold_{k}"
    # else:
    #     train_dataloader = dm.train_dataloader()
    #     val_dataloader = dm.val_dataloader()
    #     plot_file_suffix = f"_{method_name}_rep_{rep_n}"

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

    metric_1, metric_2 = get_final_val_metrics(trainer)

    model.metric1 = metric_1
    model.metric2 = metric_2

    # get reals and preds for plotting later
    # now model object has train_preds, train_reals, val_preds, val_reals

    # COMMENTING TO CHECK IF THIS IS NEEDED
    # model.get_reals_and_preds(train_dataloader, val_dataloader)

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
    trained_models_dict, data_module, params, fusion_model, init_model
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
    metric_name_list : list
        List of metric names.

    """
    if params["kfold_flag"]:
        for k in range(params["num_k"]):
            trained_model = train_and_test(
                dm=data_module,
                params=params,
                k=k,
                fusion_model=fusion_model,
                init_model=init_model,
            )
            trained_models_dict = store_trained_model(
                trained_model, trained_models_dict
            )

    else:
        trained_model = train_and_test(
            dm=data_module,
            params=params,
            k=None,
            fusion_model=fusion_model,
            init_model=init_model,
        )
        trained_models_dict = store_trained_model(trained_model, trained_models_dict)

    return trained_models_dict

    # , trainer <-- I don't think I need to return the trainer?

    # # update results
    # metric_1, metric_2 = get_final_val_metrics(metric_name_list, trainer)

    # # plot end of run figures
    # eval_figs, val_reals, val_preds = model.plot_eval_figs(
    #     train_dataloader,
    #     val_dataloader,
    #     plot_file_suffix,
    # )

    # return model, trainer, metric_1, metric_2, val_reals, val_preds


# TODO this is where we would add the code to save the model and trainer to a dictionary for later Plotter functions?
# what if it's a kfold training type?
# do we input the model and trainer dict to this function, and if it's kfold it'll know that the model key will have a list of trainers?
