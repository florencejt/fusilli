"""
This module contains classes and functions for evaluating trained models (i.e. plotting results from training).
The setup for this module has been inspired by the scikit-learn API for plotting results, which involves each plot
being a class with a ``from_final_val_data`` method that takes in a trained model and returns a plot with the validation data,
and a ``from_new_data`` method that takes in a trained model and new data and returns a plot.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from matplotlib import gridspec
from sklearn.metrics import confusion_matrix
from torch.utils.data import ConcatDataset, DataLoader

import fusilli.data as data
from fusilli.fusionmodels.base_model import BaseModel
from fusilli.utils.training_utils import (
    get_checkpoint_filename_for_trained_fusion_model,
)
from fusilli.utils import model_modifier


class ParentPlotter:
    """Parent class for all plot classes.

    It includes methods that are used by multiple plot classes, such as obtaining final
    validation data from kfold and train/test models, and putting new data through the
    models for both kfold and train/test protocols.
    """

    def __init__(self):
        pass

    @classmethod
    def get_kfold_data_from_model(cls, model_list):
        """
        Get the final validation data from a kfold model, meaning the data that was used to evaluate the model
        when the model training was complete.

        Parameters
        ----------
        model_list: list
            List of trained pytorch_lightning models. For kfold models, this is a list of at least length 2,
            where the first element is the k=1 model and the second element is the k=2 model, etc.

        Returns
        -------
        train_reals: list
            List of torch.Tensors of the real values for the training set for each fold.
            This is stored in each trained model's class instance.
        train_preds: list
            List of torch.Tensors of the predicted values for the training set for each fold.
            This is stored in each trained model's class instance.
        val_reals: list
            List of torch.Tensors of the real values for the validation set for each fold.
            This is stored in each trained model's class instance.
        val_preds: list
            List of torch.Tensors of the predicted values for the validation set for each fold.
            This is stored in each trained model's class instance.
        metrics_per_fold: dict
            Dictionary of the metrics for each fold.
            The keys are the names of the metrics and the values are lists of the metric values for each fold.
        overall_kfold_metrics: dict
            Dictionary of the overall kfold metrics.
            The keys are the names of the metrics and the values are the metric values for the overall kfold,
            meaning the metric values for the concatenated final validation data over all the folds.

        """
        train_reals = []
        train_preds = []
        val_reals = []
        val_preds = []
        val_logits = []

        metric_names = list(model_list[0].metrics.keys())

        metrics_per_fold = {}
        for metric_name in metric_names:
            metrics_per_fold[metric_name.lower()] = []

        # loop through the folds
        for fold in model_list:  # 0 is the model, 1 is the ckpt path
            # get the data points
            train_reals.append(fold.train_reals.cpu())
            train_preds.append(fold.train_preds.cpu())
            val_reals.append(fold.val_reals.cpu())
            val_preds.append(fold.val_preds.cpu())
            val_logits.append(fold.val_logits.cpu())

            # get the metrics
            for i, metric in enumerate(fold.final_val_metrics):
                metrics_per_fold[metric_names[i].lower()].append(metric)

        # concatenate the validation data points for the overall kfold performance
        all_val_reals = torch.cat(val_reals, dim=-1)
        all_val_preds = torch.cat(val_preds, dim=-1)
        all_val_logits = torch.cat(val_logits, dim=0)

        # get the overall kfold metrics
        overall_kfold_metrics = {}

        for metric_name, metric_func in model_list[0].metrics.items():
            val_metric = metric_func(
                preds=model_list[0].safe_squeeze(all_val_preds),
                labels=model_list[0].safe_squeeze(all_val_reals),
                logits=model_list[0].safe_squeeze(all_val_logits),
            )

            overall_kfold_metrics[metric_name.lower()] = val_metric.cpu().detach().item()

        return (
            train_reals,
            train_preds,
            val_reals,
            val_preds,
            metrics_per_fold,
            overall_kfold_metrics,
        )

    @classmethod
    def get_tt_data_from_model(cls, model_list):
        """
        Get the final validation data from a train/test model, meaning the data that was used to evaluate the model
        when the model training was complete.

        Parameters
        ----------
        model_list: list
            A list of length 1 containing the trained pytorch_lightning model.

        Returns
        -------
        train_reals: torch.Tensor
            Torch.Tensor of the real values for the training set.
            This is stored in the trained model's class instance.
        train_preds: torch.Tensor
            Torch.Tensor of the predicted values for the training set.
            This is stored in the trained model's class instance.
        val_reals: torch.Tensor
            Torch.Tensor of the real values for the validation set.
            This is stored in the trained model's class instance.
        val_preds: torch.Tensor
            Torch.Tensor of the predicted values for the validation set.
            This is stored in the trained model's class instance.
        metric_values: dict
            Dictionary of the metrics for the model.
            The keys are the names of the metrics and the values are the metric values for the model.

        """

        model = model_list[0]  # get the model from the list of length 1

        # not training the model
        model.eval()

        # data points
        train_reals = model.train_reals.cpu()
        train_preds = model.train_preds.cpu()
        val_reals = model.val_reals.cpu()
        val_preds = model.val_preds.cpu()

        # metrics
        metric_values = {}
        for i, metric in enumerate(model.metrics):
            metric_values[metric] = model.final_val_metrics[i]

        return train_reals, train_preds, val_reals, val_preds, metric_values

    @classmethod
    def get_new_kfold_data(
            cls, model_list, output_paths, test_data_paths, checkpoint_file_suffix=None, layer_mods=None
    ):
        """
        Get new data by running through trained model for a kfold model.

        Parameters
        ----------
        model_list: list
            List of trained pytorch_lightning models. For kfold models, this is a list of at least length 2,
            where the first element is the k=1 model and the second element is the k=2 model, etc.
        output_paths: dict
            Dictionary of the output paths. Used for knowing where the checkpoint files are stored and where to save the plots.
        test_data_paths: dict
            Dictionary of the paths to the new data. The keys are the names of the data types (e.g. "tabular1", "image").
        checkpoint_file_suffix: str, optional
            Suffix that is on the trained model checkpoint files. e.g. "_firsttry". Added by the user.
            Default is None.
        layer_mods: dict, optional
            Dictionary of the layer modifications to make to the model.

        Returns
        -------
        train_reals: list
            List of torch.Tensors of the real values for the training set for each fold.
            This is stored in each trained model's class instance.
        train_preds: list
            List of torch.Tensors of the predicted values for the training set for each fold.
            This is stored in each trained model's class instance.
        val_reals: list
            List of torch.Tensors of the real values for the new data set for each fold.
        val_preds: list
            List of torch.Tensors of the predicted values for the new data set for each fold.
            This was obtained by running the new data through the trained model.
        metrics_per_fold: dict
            Dictionary of the metrics for each fold.
            The keys are the names of the metrics and the values are lists of the metric values for each fold.
        overall_kfold_metrics: dict
            Dictionary of the overall kfold metrics.
            The keys are the names of the metrics and the values are the metric values for the overall kfold,
            meaning the metric values for the concatenated new data over all the folds.

        Raises
        ------
        ValueError
            If the model has a graph maker, it's not supported yet for creating graphs from new data.
        """

        if checkpoint_file_suffix is None:
            checkpoint_file_suffix = ""

        train_reals = []
        train_preds = []
        val_reals = []
        val_preds = []
        val_logits = []

        metric_names = list(model_list[0].metrics.keys())

        # dictionary to store the metrics for each fold
        metrics_per_fold = {}
        for metric_name in metric_names:
            metrics_per_fold[metric_name.lower()] = []

        output_paths_copy = output_paths.copy()

        num_folds = len(model_list)

        # loop through the folds and get the predictions for each fold
        for k, fold_model in enumerate(model_list):
            # eval the model

            model = fold_model
            # ckpt_path = fold_model[1]

            model.eval()

            if hasattr(model.model, "graph_maker"):
                raise ValueError(
                    "Model has a graph maker. This is not supported yet for creating graphs from new data."
                )

            if model.model.subspace_method is not None:
                subspace_ckpts = []
                for subspace_model in model.model.subspace_method.subspace_models:
                    subspace_ckpts.append(
                        output_paths["checkpoints"]
                        + "/"
                        + "subspace_"
                        + model.model.__class__.__name__
                        + "_"
                        + subspace_model.__name__
                        + "_fold_"
                        + str(k)
                        + checkpoint_file_suffix
                        + ".ckpt"
                    )

            else:
                subspace_ckpts = None

            dm = data.prepare_fusion_data(
                prediction_task=model.model.prediction_task,
                fusion_model=model.model,
                data_paths=test_data_paths,
                output_paths=output_paths_copy,
                kfold=True,
                num_folds=num_folds,
                checkpoint_path=subspace_ckpts,
                layer_mods=layer_mods,
            )

            # just taking the first fold because we don't need to split the new data into folds
            # we just wanted to convert it to latent using that fold's trained subspace model
            dm.train_dataset = dm.folds[0][0]
            dm.test_dataset = dm.folds[0][1]

            dataset = ConcatDataset([dm.train_dataset, dm.test_dataset])
            dataloader = DataLoader(dataset, batch_size=len(dataset))

            trained_fusion_model_checkpoint = (
                get_checkpoint_filename_for_trained_fusion_model(
                    checkpoint_dir=output_paths["checkpoints"],
                    model=model,
                    checkpoint_file_suffix=checkpoint_file_suffix,
                    fold=k
                )
            )

            # init model
            new_model = BaseModel(
                model=model.model.__class__(
                    prediction_task=model.model.prediction_task,
                    data_dims=dm.data_dims,  # data_dims is a list of tuples
                    multiclass_dimensions=dm.multiclass_dimensions,
                ),
                metrics_list=model.metrics_list,
            )

            # modify layers if needed
            if layer_mods is not None:
                new_model.model = model_modifier.modify_model_architecture(
                    new_model.model,
                    layer_mods,
                )
            # load the state dict
            new_model.load_state_dict(torch.load(trained_fusion_model_checkpoint)["state_dict"])

            new_model.eval()

            fold_val_preds = []
            fold_val_logits = []
            fold_val_reals = []

            for batch in dataloader:
                x, y = new_model.get_data_from_batch(batch)
                out = new_model.get_model_outputs_and_loss(x, y)
                loss, end_output, logits = out

                fold_val_preds.append(end_output.cpu().detach())
                fold_val_logits.append(logits.cpu().detach())
                fold_val_reals.append(y.cpu().detach())

            fold_val_reals = torch.cat(fold_val_reals, dim=-1)
            fold_val_preds = torch.cat(fold_val_preds, dim=-1)
            fold_val_logits = torch.cat(fold_val_logits, dim=0)

            val_reals.append(fold_val_reals)
            val_preds.append(fold_val_preds)
            val_logits.append(fold_val_logits)

            # training data points from the old trained BaseModel
            train_reals.append(model.train_reals.cpu().detach())
            train_preds.append(model.train_preds.cpu().detach())

            for metric_name, metric_func in new_model.metrics.items():
                val_step_metric = metric_func(
                    preds=model_list[0].safe_squeeze(fold_val_preds),
                    labels=model_list[0].safe_squeeze(fold_val_reals),
                    logits=model_list[0].safe_squeeze(fold_val_logits),
                )

                metrics_per_fold[metric_name.lower()].append(val_step_metric)

        # concatenate the validation data points for the overall kfold performance
        all_val_reals = torch.cat(val_reals, dim=-1)
        all_val_preds = torch.cat(val_preds, dim=-1)
        all_val_logits = torch.cat(val_logits, dim=0)

        # get the overall kfold metrics
        overall_kfold_metrics = {}

        for metric_name, metric_func in new_model.metrics.items():
            val_metric = metric_func(
                preds=model_list[0].safe_squeeze(all_val_preds),
                labels=model_list[0].safe_squeeze(all_val_reals),
                logits=model_list[0].safe_squeeze(all_val_logits),
            )

            overall_kfold_metrics[metric_name.lower()] = val_metric.cpu().detach().item()

        return (
            train_reals,
            train_preds,
            val_reals,
            val_preds,
            metrics_per_fold,
            overall_kfold_metrics,
        )

    @classmethod
    def get_new_tt_data(
            cls, model_list, output_paths, test_data_paths, checkpoint_file_suffix=None, layer_mods=None
    ):
        """
        Get new data by running through trained model for a train/test model.

        Parameters
        ----------
        model_list: list
            A list of length 1 containing the trained pytorch_lightning model.
        output_paths: dict
            Dictionary of the output paths. Used for knowing where the checkpoint files are stored and where to save the plots.
        test_data_paths: dict
            Dictionary of the paths to the new data. The keys are the names of the data types (e.g. "tabular1", "image").
        checkpoint_file_suffix: str, optional
            Suffix that is on the trained model checkpoint files. e.g. "_firsttry". Added by the user.
            Default is None.
        layer_mods: dict, optional
            Dictionary of the layer modifications to make to the model.

        Returns
        -------
        train_reals: torch.Tensor
            Torch.Tensor of the real values for the training set.
            This is stored in the trained model's class instance.
        train_preds: torch.Tensor
            Torch.Tensor of the predicted values for the training set.
            This is stored in the trained model's class instance.
        val_reals: torch.Tensor
            Torch.Tensor of the real values for the new data set.
        val_preds: torch.Tensor
            Torch.Tensor of the predicted values for the new data set.
        metric_values: dict
            Dictionary of the metrics for the model.
            The keys are the names of the metrics and the values are the metric values for the model.

        Raises
        ------
        ValueError
            If the model has a graph maker, it's not supported yet for creating graphs from new data.

        """

        if checkpoint_file_suffix is None:
            checkpoint_file_suffix = ""

        # eval the model
        # ckpt_path = model[0][1]
        model = model_list[0]

        model.eval()

        if hasattr(model.model, "graph_maker"):
            raise ValueError(
                "Model has a graph maker. This is not supported yet for creating graphs from new data."
            )

        if model.model.subspace_method is not None:
            subspace_ckpts = []
            for subspace_model in model.model.subspace_method.subspace_models:
                subspace_ckpts.append(
                    output_paths["checkpoints"]
                    + "/"
                    + "subspace_"
                    + model.model.__class__.__name__
                    + "_"
                    + subspace_model.__name__
                    + checkpoint_file_suffix
                    + ".ckpt"
                )

        else:
            subspace_ckpts = None

        # get data module (potentially will need to be trained with a subspace method or graph-maker)
        dm = data.prepare_fusion_data(
            prediction_task=model.model.prediction_task,
            fusion_model=model.model.__class__,
            data_paths=test_data_paths,
            output_paths=output_paths,
            checkpoint_path=subspace_ckpts,
            layer_mods=layer_mods,
        )

        # concatenating the train and test datasets because we want to get the predictions for all the data
        dataset = ConcatDataset([dm.train_dataset, dm.test_dataset])
        dataloader = DataLoader(dataset, batch_size=len(dataset))

        # get ckpt_path from fusion name
        trained_fusion_model_checkpoint = (
            get_checkpoint_filename_for_trained_fusion_model(
                output_paths["checkpoints"], model, checkpoint_file_suffix, fold=None
            )
        )

        # init model
        new_model = BaseModel(
            model=model.model.__class__(
                prediction_task=model.model.prediction_task,
                # prediction_task is a string (binary, regression, multiclass)
                data_dims=dm.data_dims,  # data_dims is a list of tuples
                multiclass_dimensions=dm.multiclass_dimensions,
            ),
            metrics_list=model.metrics_list,
        )

        # modify layers if needed
        if layer_mods is not None:
            new_model.model = model_modifier.modify_model_architecture(
                new_model.model,
                layer_mods,
            )
        # load the state dict
        new_model.load_state_dict(torch.load(trained_fusion_model_checkpoint)["state_dict"])

        new_model.eval()
        # get the predictions
        end_outputs_list = []
        logits_list = []
        reals_list = []

        for batch in dataloader:
            x, y = new_model.get_data_from_batch(batch)
            out = new_model.get_model_outputs_and_loss(x, y)
            loss, end_output, logits = out

            end_outputs_list.append(new_model.safe_squeeze(end_output).cpu().detach())
            logits_list.append(new_model.safe_squeeze(logits).cpu().detach())
            reals_list.append(new_model.safe_squeeze(y).cpu().detach())

        # get the train reals, train preds, val reals, val preds
        train_reals = model.train_reals.cpu()
        train_preds = model.train_preds.cpu()
        val_preds = torch.cat(end_outputs_list, dim=-1)
        val_reals = torch.cat(reals_list, dim=-1)
        val_logits = torch.cat(logits_list, dim=0)

        # get the metrics
        metric_values = {}

        for metric_name, metric_func in new_model.metrics.items():
            val_step_metric = metric_func(
                preds=new_model.safe_squeeze(val_preds),
                labels=new_model.safe_squeeze(val_reals),
                logits=new_model.safe_squeeze(val_logits),
            )
            metric_values[metric_name.lower()] = val_step_metric

        return train_reals, train_preds, val_reals, val_preds, metric_values


class RealsVsPreds(ParentPlotter):
    """
    Plots the real values vs the predicted values for a model.
    Pink dots are the training data, green dots are the validation data. The validation data is
    either new data if using from_new_data or the original validation data if using from_final_val_data.
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def from_new_data(
            cls, model_list, output_paths, test_data_paths, checkpoint_file_suffix=None, layer_mods=None
    ):
        """

        Reals vs preds plot using new data (i.e. data that was not used to train or validate the model).

        Parameters
        ----------
        model_list: list
            List of trained pytorch_lightning models. For kfold models, this is a list of at least length 2,
            where the first element is the k=1 model and the second element is the k=2 model, etc.
            For train/test models, this is a list of length 1.
        output_paths : dict
            Dictionary of the output paths. Used for knowing where the checkpoint files are stored and where to save the plots.
        test_data_paths: dict
            Dictionary of the paths to the new data. The keys are the names of the data types (e.g. "tabular1", "image").
        checkpoint_file_suffix: str, optional
            Suffix that is on the trained model checkpoint files. e.g. "_firsttry". Added by the user.
            Default is None.
        layer_mods: dict, optional
            Dictionary of the layer modifications to make to the model.

        Returns
        -------
        figure: matplotlib.pyplot.figure
            The figure of the plot.

        Raises
        ------
        ValueError
            If the model is not a list.
            If the model is a list of length > 1 but kfold_flag is False.
            If the model is a list of length 1 but kfold_flag is True.
            If the model is an empty list.

        """

        if not isinstance(model_list, list):
            raise ValueError(
                (
                    "Argument 'model_list' is not a list. "
                    "Please check the model and the function input."
                    "If you are using a train/test model, the single model must be in a list of length 1."
                )
            )

        if len(model_list) > 1:

            (
                train_reals,
                train_preds,
                val_reals,
                val_preds,
                metrics_per_fold,
                overall_kfold_metrics,
            ) = cls.get_new_kfold_data(
                model_list, output_paths, test_data_paths, checkpoint_file_suffix, layer_mods
            )

            figure = cls.reals_vs_preds_kfold(
                model_list,
                val_reals,
                val_preds,
                metrics_per_fold,
                overall_kfold_metrics,
            )

            figure.suptitle("Evaluation: External Test Data")

        elif len(model_list) == 1:

            (
                train_reals,
                train_preds,
                val_reals,
                val_preds,
                metric_values,
            ) = cls.get_new_tt_data(
                model_list, output_paths, test_data_paths, checkpoint_file_suffix, layer_mods
            )

            # plot the figure
            figure = cls.reals_vs_preds_tt(
                model_list,
                train_reals,
                train_preds,
                val_reals,
                val_preds,
                metric_values,
            )

            figure.suptitle("Evaluation: External Test Data")

        else:
            raise ValueError("Argument 'model_list' is an empty list. ")

        return figure

    @classmethod
    def from_final_val_data(cls, model_list):
        """
        Reals vs preds plot using the final validation data (i.e. the data that was used to evaluate the model
        when the model training was complete).

        Parameters
        ----------
        model_list: list
            List of trained pytorch_lightning models. For kfold models, this is a list of at least length 2, where the first
            element is the k=1 model and the second element is the k=2 model, etc.
            For train/test models, this is a list of length 1.

        Returns
        -------
        figure: matplotlib.pyplot.figure
            The figure of the plot.

        Raises
        ------
        ValueError
            If the model is not a list.
            If the model is a list of length > 1 but kfold_flag is False.
            If the model is a list of length 1 but kfold_flag is True.
            If the model is an empty list.


        """
        if not isinstance(model_list, list):
            raise ValueError(
                (
                    "Argument 'model_list' is not a list. "
                    "Please check the model and the function input."
                    "If you are using a train/test model, the single model must be in a list of length 1."
                )
            )

        if len(model_list) > 1:  # kfold model (list of models and their checkpoints)

            (
                train_reals,
                train_preds,
                val_reals,
                val_preds,
                metrics_per_fold,
                overall_kfold_metrics,
            ) = cls.get_kfold_data_from_model(model_list)

            figure = cls.reals_vs_preds_kfold(
                model_list,
                val_reals,
                val_preds,
                metrics_per_fold,
                overall_kfold_metrics,
            )

            figure.suptitle("Evaluation: Validation Data")

        elif len(model_list) == 1:

            # get the data
            (
                train_reals,
                train_preds,
                val_reals,
                val_preds,
                metric_values,
            ) = cls.get_tt_data_from_model(model_list)

            # plot the figure
            figure = cls.reals_vs_preds_tt(
                model_list,
                train_reals,
                train_preds,
                val_reals,
                val_preds,
                metric_values,
            )

            figure.suptitle("Evaluation: Validation Data")

        else:
            raise ValueError(("Argument 'model_list' is an empty list. "))

        return figure

    @classmethod
    def reals_vs_preds_kfold(
            cls,
            model_list,
            val_reals,
            val_preds,
            metrics_per_fold,
            overall_kfold_metrics,
    ):
        """
        Reals vs preds plot for a kfold model. This function should be called within the RealVsPreds class
        after the k-fold data has been obtained from the model (either old data or new data).

        Parameters
        ----------
        model_list: list
            List of trained pytorch_lightning models. For kfold models, this is a list of at least length 2, where the first
            element is the k=1 model and the second element is the k=2 model, etc.
        val_reals: list
            List of torch.Tensors of the real values for the new data set for each fold.
        val_preds: list
            List of torch.Tensors of the predicted values for the new data set for each fold.
        metrics_per_fold: dict
            Dictionary of the metrics for each fold.
            The keys are the names of the metrics and the values are lists of the metric values for each fold.
        overall_kfold_metrics: dict
            Dictionary of the overall kfold metrics.
            The keys are the names of the metrics and the values are the metric values for the overall kfold,
            meaning the metric values for the concatenated new data over all the folds.

        Returns
        -------
        fig: matplotlib.pyplot.figure
            The figure of the plot.
        """

        first_fold_model = model_list[0]
        metric_names = list(metrics_per_fold.keys())
        N = len(model_list)

        cols = 3
        rows = int(math.ceil(N / cols))

        fig = plt.figure(constrained_layout=True, figsize=(15, 6))
        subplots = fig.subfigures(1, 2)

        ax0 = subplots[0].subplots(1, 1)

        gs = gridspec.GridSpec(
            rows,
            cols,
            hspace=0.5,
            wspace=0.7,
        )

        for n in range(N):
            if n == 0:
                ax1 = subplots[1].add_subplot(gs[n])
                ax_og = ax1
            else:
                ax1 = subplots[1].add_subplot(gs[n], sharey=ax_og, sharex=ax_og)

            # get real and predicted values for the current fold
            reals = val_reals[n]
            preds = val_preds[n]

            # plot real vs. predicted values
            ax1.scatter(reals, preds, c="#f082ef", marker="o")

            # plot x=y line as a dashed line
            ax1.plot(
                [0, 1],
                [0, 1],
                color="k",
                linestyle="--",
                alpha=0.75,
                zorder=0,
                transform=ax1.transAxes,
            )

            # set title of plot to the metric for the current fold
            ax1.set_title(
                f"Fold {n + 1}: {metric_names[0]}={float(metrics_per_fold[metric_names[0].lower()][n]):.3f}"
            )

            # set x and y labels
            ax1.set_xlabel("Real Values")
            ax1.set_ylabel("Predictions")

        all_val_reals = torch.cat(val_reals, dim=-1)
        all_val_preds = torch.cat(val_preds, dim=-1)

        # plot all real vs. predicted values
        ax0.scatter(all_val_reals, all_val_preds, c="#f082ef", marker="o")

        # plot x=y line as a dashed line
        ax0.plot(
            [0, 1],
            [0, 1],
            color="k",
            linestyle="--",
            alpha=0.75,
            zorder=0,
            transform=ax0.transAxes,
        )
        ax0.set_title(
            (
                f"{first_fold_model.model.method_name}: {metric_names[0]}"
                f"={float(overall_kfold_metrics[metric_names[0]]):.3f}"
            )
        )

        # set x and y labels
        ax0.set_xlabel("Real Values")
        ax0.set_ylabel("Predictions")

        # Set the overall title for the entire figure
        fig.suptitle(
            f"{first_fold_model.model.__class__.__name__}: reals vs. predicteds"
        )

        return fig

    @classmethod
    def reals_vs_preds_tt(
            cls, model_list, train_reals, train_preds, val_reals, val_preds, metric_values
    ):
        """
        Reals vs preds plot for a train/test model. This function should be called within the RealVsPreds class
        after the train/test data has been obtained from the model (either old data or new data).

        Parameters
        ----------
        model_list: list
            A list of length 1 containing the trained pytorch_lightning model.
        train_reals: torch.Tensor
            Torch.Tensor of the real values for the training set.
        train_preds: torch.Tensor
            Torch.Tensor of the predicted values for the training set.
        val_reals: torch.Tensor
            Torch.Tensor of the real values for the new data set.
        val_preds: torch.Tensor
            Torch.Tensor of the predicted values for the new data set.
        metric_values: dict
            Dictionary of the metrics for the model.
            The keys are the names of the metrics and the values are the metric values for the model.

        Returns
        -------
        fig: matplotlib.pyplot.figure
            The figure of the plot.

        """

        model = model_list[0]

        fig, ax = plt.subplots()

        ax.scatter(
            train_reals,
            train_preds,
            c="#f082ef",
            marker="o",
            label="Train",
        )
        ax.scatter(val_reals, val_preds, c="#00b64e", marker="^", label="Validation")

        # Get the limits of the current scatter plot
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()

        # Set up data points for the x=y line
        line_x = np.linspace(min(x_min, y_min), max(x_max, y_max), 100)
        line_y = line_x

        # Plot the x=y line as a dashed line
        plt.plot(line_x, line_y, linestyle="dashed", color="black", label="x=y Line")

        metric1_name = list(metric_values.keys())[0]
        ax.set_title(
            (
                f"{model.model.method_name} - Validation {metric1_name}:"
                f" {float(metric_values[metric1_name]):.3f}"
            )
        )

        ax.set_xlabel("Real Values")
        ax.set_ylabel("Predictions")
        ax.legend()

        return fig


class ConfusionMatrix(ParentPlotter):
    """
    Plots the confusion matrix for a model. This should be used for classification models only (binary or multiclass).
    The data used to create the confusion matrix is either new data if using from_new_data or the original validation data
    if using from_final_val_data.
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def from_new_data(cls, model_list, output_paths, test_data_paths,
                      checkpoint_file_suffix=None, layer_mods=None):
        """
        Confusion matrix using new data (i.e. data that was not used to train or validate the model).

        Parameters
        ----------
        model_list: list
            List of trained pytorch_lightning models. For kfold models, this is a list of at least length 2,
            where the first element is the k=1 model and the second element is the k=2 model, etc.
            For train/test models, this is a list of length 1.
        output_paths: dict
            Dictionary of the output paths. Used for knowing where the checkpoint files are stored and where to save the plots.
        test_data_paths: dict
            Dictionary of the paths to the new data. The keys are the names of the data types (e.g. "tabular1", "image").
        checkpoint_file_suffix: str, optional
            Suffix that is on the trained model checkpoint files. e.g. "_firsttry". Added by the user.

        Returns
        -------
        figure: matplotlib.pyplot.figure
            The figure of the plot.

        Raises
        ------
        ValueError
            If the model is not a list.
            If the model is a list of length > 1 but kfold_flag is False.
            If the model is a list of length 1 but kfold_flag is True.
            If the model is an empty list.

        """

        if not isinstance(model_list, list):
            raise ValueError(
                (
                    "Argument 'model_list' is not a list. "
                    "Please check the model and the function input."
                    "If you are using a train/test model, the single model must be in a list of length 1."
                )
            )

        if len(model_list) > 1:  # kfold model

            (
                train_reals,
                train_preds,
                val_reals,
                val_preds,
                metrics_per_fold,
                overall_kfold_metrics,
            ) = cls.get_new_kfold_data(model_list, output_paths, test_data_paths, checkpoint_file_suffix, layer_mods)

            figure = cls.confusion_matrix_kfold(
                model_list,
                val_reals,
                val_preds,
                metrics_per_fold,
                overall_kfold_metrics,
            )

        elif len(model_list) == 1:  # train/test model

            (
                train_reals,
                train_preds,
                val_reals,
                val_preds,
                metric_values,
            ) = cls.get_new_tt_data(model_list, output_paths, test_data_paths, checkpoint_file_suffix,
                                    layer_mods)

            # plot the figure
            figure = cls.confusion_matrix_tt(
                model_list, val_reals, val_preds, metric_values
            )

        else:
            raise ValueError("Argument 'model_list' is an empty list. ")

        return figure

    @classmethod
    def from_final_val_data(cls, model_list):
        """
        Confusion matrix using the final validation data (i.e. the data that was used to evaluate the model
        when the model training was complete).

        Parameters
        ----------
        model_list: list
            List of trained pytorch_lightning models. For kfold models, this is a list of at least length 2, where the first
            element is the k=1 model and the second element is the k=2 model, etc.
            For train/test models, this is a list of length 1.

        Returns
        -------
        figure: matplotlib.pyplot.figure
            The figure of the plot.

        Raises
        ------
        ValueError
            If the model is not a list.
            If the model is a list of length > 1 but kfold_flag is False.
            If the model is a list of length 1 but kfold_flag is True.
            If the model is an empty list.
        """
        if not isinstance(model_list, list):
            raise ValueError(
                (
                    "Argument 'model_list' is not a list. "
                    "Please check the model and the function input."
                    "If you are using a train/test model, the single model must be in a list of length 1."
                )
            )

        if len(model_list) > 1:  # kfold model

            (
                train_reals,
                train_preds,
                val_reals,
                val_preds,
                metrics_per_fold,
                overall_kfold_metrics,
            ) = cls.get_kfold_data_from_model(model_list)

            figure = cls.confusion_matrix_kfold(
                model_list,
                val_reals,
                val_preds,
                metrics_per_fold,
                overall_kfold_metrics,
            )

        elif len(model_list) == 1:  # train/test model

            (
                train_reals,
                train_preds,
                val_reals,
                val_preds,
                metric_values,
            ) = cls.get_tt_data_from_model(model_list)

            figure = cls.confusion_matrix_tt(
                model_list, val_reals, val_preds, metric_values
            )

        else:
            raise ValueError(("Argument 'model_list' is an empty list. "))

        return figure

    @classmethod
    def confusion_matrix_tt(cls, model_list, val_reals, val_preds, metric_values):
        """
        Confusion matrix for a train/test model. This function should be called within the ConfusionMatrix class
        after the train/test data has been obtained from the model (either old data or new data).

        Parameters
        ----------
        model_list: list
            A list of length 1 containing the trained pytorch_lightning model.
        val_reals: torch.Tensor
            Torch.Tensor of the real values for the new data set.
        val_preds: torch.Tensor
            Torch.Tensor of the predicted values for the new data set.
        metric_values: dict
            Dictionary of the metrics for the model.
            The keys are the names of the metrics and the values are the metric values for the model.

        Returns
        -------
        fig: matplotlib.pyplot.figure
            The figure of the plot.
        """
        conf_matrix = confusion_matrix(y_true=val_reals, y_pred=val_preds)

        # Create a figure and axis for the plot
        fig, ax = plt.subplots(figsize=(7.5, 7.5))

        # Plot the confusion matrix as a heatmap
        ax.matshow(conf_matrix, cmap=plt.cm.RdPu, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                # Add the value of each cell to the plot
                ax.text(
                    x=j,
                    y=i,
                    s=conf_matrix[i, j],
                    va="center",
                    ha="center",
                    size="xx-large",
                )

        plt.xlabel("Predictions", fontsize=18)
        plt.ylabel("Actuals", fontsize=18)

        metric1_name = list(metric_values.keys())[0]

        plt.title(
            f"{model_list[0].model.method_name} - Validation {metric1_name}: {float(metric_values[metric1_name]):.3f}"
        )

        plt.tight_layout()

        return fig

    @classmethod
    def confusion_matrix_kfold(
            cls,
            model_list,
            val_reals,
            val_preds,
            metrics_per_fold,
            overall_kfold_metrics,
    ):
        """
        Confusion matrix for a kfold model. This function should be called within the ConfusionMatrix class
        after the k-fold data has been obtained from the model (either old data or new data).

        Parameters
        ----------
        model_list: list
            List of trained pytorch_lightning models. For kfold models, this is a list of at least length 2, where the first
            element is the k=1 model and the second element is the k=2 model, etc.
        val_reals: list
            List of torch.Tensors of the real values for the new data set for each fold.
        val_preds: list
            List of torch.Tensors of the predicted values for the new data set for each fold.
        metrics_per_fold: dict
            Dictionary of the metrics for each fold.
            The keys are the names of the metrics and the values are lists of the metric values for each fold.
        overall_kfold_metrics: dict
            Dictionary of the overall kfold metrics.
            The keys are the names of the metrics and the values are the metric values for the overall kfold,
            meaning the metric values for the concatenated new data over all the folds.

        Returns
        -------
        fig: matplotlib.pyplot.figure
            The figure of the plot.

        """
        first_fold_model = model_list[0]
        metric_names = list(metrics_per_fold.keys())
        N = len(model_list)

        cols = 3
        rows = int(math.ceil(N / cols))

        fig = plt.figure(constrained_layout=True, figsize=(15, 6))
        subplots = fig.subfigures(1, 2)

        ax0 = subplots[0].subplots(1, 1)

        gs = gridspec.GridSpec(
            rows,
            cols,
            hspace=0.5,
            wspace=0.5,
        )

        for n in range(N):
            if n == 0:
                ax1 = subplots[1].add_subplot(gs[n])
                ax_og = ax1
            else:
                ax1 = subplots[1].add_subplot(gs[n], sharey=ax_og, sharex=ax_og)

            # get real and predicted values for the current fold
            reals = val_reals[n]
            preds = val_preds[n]

            conf_matrix = confusion_matrix(y_true=reals, y_pred=preds.squeeze())
            ax1.matshow(conf_matrix, cmap=plt.cm.RdPu, alpha=0.5)

            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    # Add the value of each cell to the plot
                    ax1.text(
                        x=j,
                        y=i,
                        s=conf_matrix[i, j],
                        va="center",
                        ha="center",
                        size="large",
                    )

            ax1.set_xlabel("Predictions", fontsize=10)
            ax1.set_ylabel("Actuals", fontsize=10)

            ax1.set_title(
                f"Fold {n + 1}:\n{metric_names[0]}={float(metrics_per_fold[metric_names[0].lower()][n]):.3f}"
            )

        # gs.tight_layout(fig)
        all_val_reals = torch.cat(val_reals, dim=-1)
        all_val_preds = torch.cat(val_preds, dim=-1)

        # plot all real vs. predicted values
        conf_matrix = confusion_matrix(
            y_true=all_val_reals, y_pred=all_val_preds.squeeze()
        )

        # Plot the confusion matrix as a heatmap
        ax0.matshow(conf_matrix, cmap=plt.cm.RdPu, alpha=0.3)

        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                # Add the value of each cell to the plot
                ax0.text(
                    x=j,
                    y=i,
                    s=conf_matrix[i, j],
                    va="center",
                    ha="center",
                    size="xx-large",
                )

        ax0.set_xlabel("Predictions", fontsize=18)
        ax0.set_ylabel("Actuals", fontsize=18)

        ax0.set_title(
            (
                f"{first_fold_model.model.method_name}: {metric_names[0]}"
                f"={float(overall_kfold_metrics[metric_names[0]]):.3f}"
            )
        )

        # Set the overall title for the entire figure
        fig.suptitle(f"{first_fold_model.model.__class__.__name__}: confusion matrix")

        return fig


class ModelComparison(ParentPlotter):
    """
    Plots the performance of multiple models on a single plot. Currently (as of 2023-10-11) this is only
    implemented for from_final_val_data because it is not clear how to implement from_new_data for graph-based models, so
    they would have to be left out of the main plot (which feels wrong tbh).

    """

    def __init__(self):
        super().__init__()

    @classmethod
    def from_final_val_data(cls, model_dict):
        """
        Plotting function for comparing models on metrics using the final validation data (i.e. the data that was used to evaluate the model
        when the model training was complete).
        Produces a violin plot if kfold_flag is True and a bar plot if kfold_flag is False.

        Parameters
        ----------
        model_dict: dict
            Dictionary of trained pytorch_lightning models.
            Keys are the names of the models and values are lists of the trained pytorch_lightning models.
            If kfold_flag is True, the lists must be of length > 1 (and the length of num_k)
            If kfold_flag is False, the lists must be of length 1 (meaning there is only one model for each key).

        Returns
        -------
        fig: matplotlib.pyplot.figure
            The figure of the plot.
        df: pandas.DataFrame
            The dataframe of the metrics.


        """
        # error if model_dict isn't a dict
        if not isinstance(model_dict, dict):
            raise ValueError(
                (
                    "Argument 'model_dict' is not a dict. "
                    "'model_dict' should have keys as the model names and values as lists of trained models. "
                )
            )

        comparing_models_metrics = {}

        # check for empty lists in dict
        for model in model_dict:
            model_list = model_dict[model]
            if len(model_list) == 0:
                raise ValueError(
                    (
                        "Empty list of models has been passed into the ModelComparison.from_final_val_data. "
                        "There is an empty list somewhere in the model_dict. Please check the model_dict."
                    )
                )

        # get kfold_flag from first model in model_dict
        first_model = list(model_dict.values())[0][0]
        first_list_of_models = list(model_dict.values())[0]
        if len(first_list_of_models) > 1:
            kfold = True
        else:
            kfold = False

        if kfold:
            overall_kfold_metrics_dict = {}
            for model in model_dict:
                model_list = model_dict[model]  # list of length k of trained models

                # error if list is of length 1
                if len(model_list) == 1:
                    raise ValueError(
                        (
                            "List of models in model_dict has length 1 but the kfold_flag is True. "
                            "K-fold training should produce a list of models of length > 1. "
                            "Please check the model_dict and the kfold_flag."
                        )
                    )

                model_method_name = model_list[0].model.method_name

                (
                    train_reals,
                    train_preds,
                    val_reals,
                    val_preds,
                    metrics_per_fold,
                    overall_kfold_metrics,
                ) = cls.get_kfold_data_from_model(model_list)

                comparing_models_metrics[model_method_name] = metrics_per_fold

                overall_kfold_metrics_dict[model_method_name] = overall_kfold_metrics

            figure = cls.kfold_comparison_plot(comparing_models_metrics)

            df = cls.get_performance_dataframe(
                comparing_models_metrics, overall_kfold_metrics_dict, kfold
            )

        else:
            for model in model_dict:
                model_list = model_dict[model]  # list of length 1 of trained model

                # error if list is of length > 1
                if len(model_list) > 1:
                    raise ValueError(
                        (
                            "List of models in model_dict has length > 1 but the kfold_flag is False. "
                            "Train/test training should produce a list of models of length 1. "
                            "Please check the model_dict and the kfold_flag."
                        )
                    )
                model_method_name = model_list[0].model.method_name
                (
                    train_reals,
                    train_preds,
                    val_reals,
                    val_preds,
                    metric_values,
                ) = cls.get_tt_data_from_model(model_list)

                comparing_models_metrics[model_method_name] = metric_values

            figure = cls.train_test_comparison_plot(comparing_models_metrics)
            df = cls.get_performance_dataframe(
                comparing_models_metrics, None, kfold
            )

        return figure, df

    @classmethod
    def from_new_data(cls, model_dict, output_paths, test_data_paths, checkpoint_file_suffix=None,
                      layer_mods=None):
        """
        Plotting function for comparing models on metrics using new data (i.e. data that was not used to train or validate the model).
        Produces a violin plot if kfold_flag is True and a bar plot if kfold_flag is False.

        Parameters
        ----------
        model_dict: dict
            Dictionary of trained pytorch_lightning models.
            Keys are the names of the models and values are lists of the trained pytorch_lightning models.
            If kfold_flag is True, the lists must be of length > 1 (and the length of num_k)
            If kfold_flag is False, the lists must be of length 1 (meaning there is only one model for each key).
        output_paths: dict
            Dictionary of the output paths. Used for knowing where the checkpoint files are stored and where to save the plots.
        test_data_paths: dict
            Dictionary of the paths to the new data. The keys are the names of the data types (e.g. "tabular1", "image").
        checkpoint_file_suffix: str, optional
            Suffix that is on the trained model checkpoint files. e.g. "_firsttry". Added by the user.
            Default is None.
        layer_mods: dict, optional
            Dictionary of the layer modifications to make to the model.

        Returns
        -------
        fig: matplotlib.pyplot.figure
            The figure of the plot.
        df: pandas.DataFrame
            The dataframe of the metrics.
        """
        comparing_models_metrics = {}

        if not isinstance(model_dict, dict):
            raise ValueError(
                (
                    "Argument 'model_dict' is not a dict. "
                    "'model_dict' should have keys as the model names and values as lists of trained models. "
                )
            )

        # check for empty lists in dict
        for model in model_dict:
            model_list = model_dict[model]
            if len(model_list) == 0:
                raise ValueError(
                    (
                        "Empty list of models has been passed into the ModelComparison.from_new_data. "
                        "There is an empty list somewhere in the model_dict. Please check the model_dict."
                    )
                )

        # get kfold_flag from first model in model_dict
        first_model = list(model_dict.values())[0][0]
        first_model_list = list(model_dict.values())[0]
        if len(first_model_list) > 1:
            kfold = True
        else:
            kfold = False

        if kfold:
            overall_kfold_metrics_dict = {}

            for model in model_dict:
                model_list = model_dict[model]  # list of length k of trained models

                # if model is a graph-based model, skip it
                if hasattr(model_list[0].model, "graph_maker"):
                    raise Warning(
                        (
                            "Graph-based models are not currently supported for the ModelComparison.from_new_data. "
                            "The graph-based models will be skipped."
                        )
                    )
                    continue

                if len(model_list) == 1:
                    raise ValueError(
                        (
                            "List of models in model_dict has length 1 but the kfold_flag is True. "
                            "K-fold training should produce a list of models of length > 1. "
                            "Please check the model_dict and the kfold_flag."
                        )
                    )

                model_method_name = model_list[0].model.method_name

                (
                    train_reals,
                    train_preds,
                    val_reals,
                    val_preds,
                    metrics_per_fold,
                    overall_kfold_metrics,
                ) = cls.get_new_kfold_data(model_list, output_paths, test_data_paths, checkpoint_file_suffix,
                                           layer_mods)

                comparing_models_metrics[model_method_name] = metrics_per_fold

                overall_kfold_metrics_dict[model_method_name] = overall_kfold_metrics

            figure = cls.kfold_comparison_plot(comparing_models_metrics)

            df = cls.get_performance_dataframe(
                comparing_models_metrics, overall_kfold_metrics_dict, kfold
            )

        else:
            for model in model_dict:
                model_list = model_dict[model]  # list of length 1 of trained model

                # if model is a graph-based model, skip it
                if hasattr(model_list[0].model, "graph_maker"):
                    raise Warning(
                        (
                            "Graph-based models are not currently supported for the ModelComparison.from_new_data. "
                            "The graph-based models will be skipped."
                        )
                    )
                    continue

                if len(model_list) > 1:
                    raise ValueError(
                        (
                            "List of models in model_dict has length > 1 but the kfold_flag is False. "
                            "Train/test training should produce a list of models of length 1. "
                            "Please check the model_dict and the kfold_flag."
                        )
                    )

                model_method_name = model_list[0].model.method_name

                (
                    train_reals,
                    train_preds,
                    val_reals,
                    val_preds,
                    metric_values,
                ) = cls.get_new_tt_data(model_list, output_paths, test_data_paths, checkpoint_file_suffix, layer_mods)

                comparing_models_metrics[model_method_name] = metric_values

            figure = cls.train_test_comparison_plot(comparing_models_metrics)
            df = cls.get_performance_dataframe(
                comparing_models_metrics, None, kfold
            )

        return figure, df

    @classmethod
    def kfold_comparison_plot(cls, comparing_models_metrics):
        """Plotting function for comparing models on kfold metrics. Produces a violin plot.

        Parameters
        ----------
        comparing_models_metrics: dict
            Dictionary of metrics. Keys are the model names and values are dict {metric1name: metric_value, metric2name: metric_value}

        Returns
        -------
        fig: matplotlib figure
            Figure containing the violin plots.
        """
        # get method names and metric names
        method_names = list(
            comparing_models_metrics.keys()
        )  # [method1name, method2name,...]

        metric1name = list(comparing_models_metrics[method_names[0]].keys())[0]
        metric2name = list(comparing_models_metrics[method_names[0]].keys())[1]

        # get metric values for each method
        metric_1_values = [
            comparing_models_metrics[method][metric1name] for method in method_names
        ]
        metric_2_values = [
            comparing_models_metrics[method][metric2name] for method in method_names
        ]

        # Calculate mean or median of metric_1_values for sorting
        metric_1_means = np.array(metric_1_values).mean(
            axis=1
        )  # Change to median if needed

        sorted_indices = np.argsort(metric_1_means)

        # Reorder method names, metric values, and other related data
        method_names = np.array(method_names)[sorted_indices]
        metric_1_values = np.array(metric_1_values)[sorted_indices].transpose()
        metric_2_values = np.array(metric_2_values)[sorted_indices].transpose()

        # create figure 1x2 subplots
        # create figure 1x2 subplots
        fig, ax = plt.subplots(1, 2)
        ax[0].grid()
        ax[1].grid()

        # create violin plots for each metric
        bp = ax[0].violinplot(metric_1_values, vert=False, showmeans=True)

        def set_violin_colors(instance, colour):
            for pc in instance["bodies"]:
                pc.set_facecolor(colour)
                pc.set_edgecolor("black")
                pc.set_alpha(0.5)
            instance["cmeans"].set_edgecolor("black")
            instance["cmins"].set_edgecolor("black")
            instance["cmaxes"].set_edgecolor("black")
            instance["cbars"].set_edgecolor("black")

        set_violin_colors(bp, "violet")

        ax[0].yaxis.set_ticks(np.arange(len(method_names)) + 1)
        ax[0].set_yticklabels(method_names)
        ax[0].get_xaxis().tick_bottom()
        ax[0].set_xlim(right=1.0)

        bp2 = ax[1].violinplot(metric_2_values, vert=False, showmeans=True)
        set_violin_colors(bp2, "powderblue")

        ax[1].yaxis.set_ticks(np.arange(len(method_names)) + 1)
        ax[1].set_yticklabels([] * len(metric_2_values))
        ax[1].get_xaxis().tick_bottom()

        # set titles and limits
        ax[0].set_title(metric1name)
        ax[1].set_title(metric2name)
        ax[1].set_xlim(left=0.0)

        plt.suptitle("Distribution of metrics between cross-validation folds")

        plt.tight_layout()

        return fig

    @classmethod
    def train_test_comparison_plot(cls, comparing_models_metrics):
        """Plotting function for comparing models on train and test metrics. Produces a horizontal bar chart.

        Parameters
        ----------
        comparing_models_metrics: dict
            Dictionary of metrics. Keys are the model names and values are dict {metric1name: metric_value, metric2name: metric_value}


        Returns
        -------
        fig: matplotlib figure
            Figure containing the horizontal bar chart.
        """

        method_names = list(
            comparing_models_metrics.keys()
        )  # [method1name, method2name,...]

        metric1name = list(comparing_models_metrics[method_names[0]].keys())[0]
        metric2name = list(comparing_models_metrics[method_names[0]].keys())[1]

        # get metric values for each method
        metric_1_values = [
            comparing_models_metrics[method][metric1name] for method in method_names
        ]
        metric_2_values = [
            comparing_models_metrics[method][metric2name] for method in method_names
        ]

        sorted_indices = np.argsort(metric_1_values)
        method_names = np.array(method_names)[sorted_indices]
        metric_1_values = np.array(metric_1_values)[sorted_indices]
        metric_2_values = np.array(metric_2_values)[sorted_indices]

        # Create an array of indices for the x-axis
        y_indices = np.arange(len(method_names))

        # Width of the bars
        bar_width = 0.35

        # Create the figure and the primary y-axis
        fig, ax = plt.subplots(1, 2)
        ax[0].grid()
        ax[1].grid()

        # Create the first bar chart using the primary y-axis (ax1)
        bars1 = ax[0].barh(
            y_indices, metric_1_values, bar_width, color="violet", edgecolor="purple"
        )

        # black dashed line at x=0
        ax[0].axvline(x=0, color="black", linestyle="--", alpha=0.5)

        ax[0].yaxis.set_ticks(np.arange(len(method_names)))
        ax[0].set_yticklabels(method_names)
        ax[0].get_xaxis().tick_bottom()
        ax[0].set_xlim(right=1.0)

        # Create a secondary y-axis for the second metric
        # ax2 = ax1.twiny()

        # Create the second bar chart using the secondary y-axis (ax2)
        bars2 = ax[1].barh(
            y_indices,
            metric_2_values,
            bar_width,
            color="powderblue",
            edgecolor="steelblue",
        )

        ax[1].yaxis.set_ticks(np.arange(len(method_names)))
        ax[1].set_yticklabels([] * len(metric_2_values))
        ax[1].get_xaxis().tick_bottom()

        # set titles and limits
        ax[0].set_title(metric1name)
        ax[1].set_title(metric2name)
        ax[1].set_xlim(left=0.0)

        # Show the plot
        plt.suptitle("Model Performance Comparison")
        plt.tight_layout()

        return fig

    @classmethod
    def get_performance_dataframe(
            cls, comparing_models_metrics, overall_kfold_metrics_dict, kfold_flag
    ):
        """
        Get a dataframe of the performance metrics for each model.
        For kfold models, the dataframe contains the overall kfold metrics and the metrics for each fold.
        For train/test models, the dataframe contains the metrics for the train and test sets.

        Parameters
        ----------
        comparing_models_metrics: dict
            Dictionary of metrics. Keys are the model names and values are dict {metric1name: metric_value, metric2name: metric_value}.
            Metric values can be float or int.
        overall_kfold_metrics_dict: dict
            Dictionary of overall kfold metrics. Keys are the model names and values are dict {metric1name: metric_value, metric2name: metric_value}.
            This is only needed for kfold models.
        kfold_flag: bool
            True if the model is a kfold model, False if the model is a train/test model.

        Returns
        -------
        df: pandas.DataFrame
            Dataframe of the performance metrics for each model.
        """
        method_names = list(
            comparing_models_metrics.keys()
        )  # [method1name, method2name,...]
        metric_names = list(comparing_models_metrics[method_names[0]].keys())

        if kfold_flag:
            overall_kfold_metrics_copy = overall_kfold_metrics_dict.copy()

            df = pd.DataFrame(overall_kfold_metrics_copy).transpose()

            # Create a DataFrame for overall kfold metrics
            folds_df = pd.DataFrame(comparing_models_metrics).T.reset_index()
            folds_df.rename(columns={"index": "Method"}, inplace=True)

            num_folds = len(folds_df[metric_names[0]][0])

            for metric_name in metric_names:
                fold_columns = [
                    f"fold{i + 1}_{metric_name}" for i in range(num_folds)
                ]

                for i, col in enumerate(fold_columns):
                    folds_df[fold_columns[i]] = folds_df[metric_name].apply(
                        lambda x: x[i] if len(x) > i else None
                    )

            folds_df.drop(columns=metric_names, inplace=True)
            folds_df.set_index("Method", inplace=True)

            final_df = pd.concat([df, folds_df], axis=1)

            # rename final_df index to "method"
            final_df.rename_axis("Method", inplace=True)
            return final_df

        else:
            # Reshape the data into a list of dictionaries
            reshaped_data = []
            for method, metrics in comparing_models_metrics.items():
                reshaped_data.append({"Method": method, **metrics})

            # Create a DataFrame from the reshaped data
            df = pd.DataFrame(reshaped_data)
            # rename index to string"method"
            df.set_index("Method", inplace=True)

            return df
