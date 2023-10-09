"""
Functions for evaluating the performance of the models and plotting the results.
"""

import math
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch_geometric as pyg
from matplotlib import gridspec
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from torch.utils.data import ConcatDataset, DataLoader

import fusilli.data as data
from fusilli.fusion_models.base_model import BaseModel
from fusilli.utils.training_utils import (
    get_checkpoint_filename_for_trained_fusion_model,
)


class ParentPlotter:
    """Parent class for all plot classes."""

    def __init__(self):
        pass

    @classmethod
    def get_kfold_data_from_model(self, model_list):
        # create empty lists for the folds of train_reals, train_preds, val_reals, val_preds
        train_reals = []
        train_preds = []
        val_reals = []
        val_preds = []
        val_logits = []

        metric_names = [
            model_list[0].metrics[model_list[0].model.pred_type][i]["name"]
            for i in range(2)
        ]

        # dictionary to store the metrics for each fold
        metrics_per_fold = {metric_names[0]: [], metric_names[1]: []}

        # loop through the folds
        for fold in model_list:  # 0 is the model, 1 is the ckpt path
            # get the data points
            train_reals.append(fold.train_reals.cpu())
            train_preds.append(fold.train_preds.cpu())
            val_reals.append(fold.val_reals.cpu())
            val_preds.append(fold.val_preds.cpu())
            val_logits.append(fold.val_logits.cpu())

            # get the metrics
            metrics_per_fold[metric_names[0]].append(fold.metric1)
            metrics_per_fold[metric_names[1]].append(fold.metric2)

        # concatenate the validation data points for the overall kfold performance
        all_val_reals = torch.cat(val_reals, dim=-1)
        all_val_preds = torch.cat(val_preds, dim=-1)
        all_val_logits = torch.cat(val_logits, dim=0)

        # get the overall kfold metrics
        overall_kfold_metrics = {}

        for metric in model_list[0].metrics[
            model_list[0].model.pred_type
        ]:  # loop through the metrics
            if "auroc" in metric["name"]:
                predicted = all_val_logits  # AUROC needs logits
            else:
                predicted = all_val_preds

            val_step_acc = metric["metric"](
                model_list[0].safe_squeeze(predicted),
                model_list[0].safe_squeeze(all_val_reals),
            )

            overall_kfold_metrics[metric["name"]] = val_step_acc

        return (
            train_reals,
            train_preds,
            val_reals,
            val_preds,
            metrics_per_fold,
            overall_kfold_metrics,
        )

    @classmethod
    def get_tt_data_from_model(self, model):
        """
        Get the data from a train/test model.

        Parameters
        ----------
        model: nn.Module
            The trained model.

        Returns
        -------
        train_reals: torch.Tensor
            The real values for the training set.
        train_preds: torch.Tensor
            The predicted values for the training set.
        val_reals: torch.Tensor
            The real values for the validation set.
        val_preds: torch.Tensor
            The predicted values for the validation set.
        metric_values: dict
            The values of the metrics for the model.
        """

        model = model[0]

        # not training the model
        model.eval()

        # data points
        train_reals = model.train_reals.cpu()
        train_preds = model.train_preds.cpu()
        val_reals = model.val_reals.cpu()
        val_preds = model.val_preds.cpu()

        # metrics
        metric_values = {
            model.metrics[model.model.pred_type][0]["name"]: model.metric1,
            model.metrics[model.model.pred_type][1]["name"]: model.metric2,
        }

        return train_reals, train_preds, val_reals, val_preds, metric_values

    @classmethod
    def get_new_kfold_data(
        self, model_list, params, data_file_suffix, checkpoint_file_suffix=None
    ):
        """
        Putting new data into each k-fold trained model: we don't need to split the new data into folds.
        We just need to get the predictions for each fold and plot them.
        """

        train_reals = []
        train_preds = []
        val_reals = []
        val_preds = []
        val_logits = []

        metric_names = [
            model_list[0].metrics[model_list[0].model.pred_type][i]["name"]
            for i in range(2)
        ]

        # dictionary to store the metrics for each fold
        metrics_per_fold = {metric_names[0]: [], metric_names[1]: []}

        params_copy = params.copy()

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
                        params["checkpoint_dir"]
                        + "/"
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

            dm = data.get_data_module(
                model.model,
                params_copy,
                optional_suffix=data_file_suffix,
                checkpoint_path=subspace_ckpts,
            )

            # just taking the first fold because we don't need to split the new data into folds
            # we just wanted to convert it to latent using that fold's trained subspace model
            dm.train_dataset = dm.folds[0][0]
            dm.test_dataset = dm.folds[0][1]

            dataset = ConcatDataset([dm.train_dataset, dm.test_dataset])
            dataloader = DataLoader(dataset, batch_size=len(dataset))

            trained_fusion_model_checkpoint = (
                get_checkpoint_filename_for_trained_fusion_model(
                    params, model, checkpoint_file_suffix, fold=k
                )
            )

            new_model = BaseModel.load_from_checkpoint(
                trained_fusion_model_checkpoint,
                model=model.model.__class__(
                    pred_type=params[
                        "pred_type"
                    ],  # pred_type is a string (binary, regression, multiclass)
                    data_dims=dm.data_dims,  # data_dims is a list of tuples
                    params=params,  # params is a dict))
                ),
            )

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
            train_reals.append(model.train_reals.cpu().detach())
            train_preds.append(model.train_preds.cpu().detach())

            for metric in model.metrics[model.model.pred_type]:  # loop
                if "auroc" in metric["name"]:
                    predicted = fold_val_logits
                else:
                    predicted = fold_val_preds

                val_step_acc = metric["metric"](
                    model.safe_squeeze(predicted),
                    model.safe_squeeze(fold_val_reals),
                )

                metrics_per_fold[metric["name"]].append(val_step_acc)

        # concatenate the validation data points for the overall kfold performance
        all_val_reals = torch.cat(val_reals, dim=-1)
        all_val_preds = torch.cat(val_preds, dim=-1)
        all_val_logits = torch.cat(val_logits, dim=0)

        # get the overall kfold metrics
        overall_kfold_metrics = {}

        for metric in model_list[0].metrics[
            model_list[0].model.pred_type
        ]:  # loop through the metrics
            if "auroc" in metric["name"]:
                predicted = all_val_logits
            else:
                predicted = all_val_preds

            val_step_acc = metric["metric"](
                model_list[0].safe_squeeze(predicted),
                model_list[0].safe_squeeze(all_val_reals),
            )

            overall_kfold_metrics[metric["name"]] = val_step_acc

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
        self, model, params, data_file_suffix, checkpoint_file_suffix=None
    ):
        """
        Get new data by running through trained model for a train/test model.

        Parameters
        ----------
        model: nn.Module
            The trained model.
        sources: list
            List of sources to get data from. [tabular1source csv, tabular2source csv, image csv]
        params: dict
            Additional parameters.
        data_file_suffix: str
            Suffix that is on the new data csv and pt files. e.g. "_new_data" or "_test"
        checkpoint_file_suffix: str, optional
            Suffix that is on the trained model checkpoint files. e.g. "_firsttry". Added by the user.
            Default is None.

        Returns
        -------
        return train_reals, train_preds, val_reals, val_preds, metric_values
        """

        # eval the model
        # ckpt_path = model[0][1]
        model = model[0]

        model.eval()

        if hasattr(model.model, "graph_maker"):
            raise ValueError(
                "Model has a graph maker. This is not supported yet for creating graphs from new data."
            )

        if model.model.subspace_method is not None:
            subspace_ckpts = []
            for subspace_model in model.model.subspace_method.subspace_models:
                subspace_ckpts.append(
                    params["checkpoint_dir"]
                    + "/"
                    + model.model.__class__.__name__
                    + "_"
                    + subspace_model.__name__
                    + checkpoint_file_suffix
                    + ".ckpt"
                )

        else:
            subspace_ckpts = None

        # get data module (potentially will need to be trained with a subspace method or graph-maker)
        dm = data.get_data_module(
            model.model.__class__,
            params,
            optional_suffix=data_file_suffix,
            checkpoint_path=subspace_ckpts,
        )
        # concatenating the train and test datasets because we want to get the predictions for all the data
        dataset = ConcatDataset([dm.train_dataset, dm.test_dataset])
        dataloader = DataLoader(dataset, batch_size=len(dataset))

        # get ckpt_path from fusion name
        trained_fusion_model_checkpoint = (
            get_checkpoint_filename_for_trained_fusion_model(
                params, model, checkpoint_file_suffix, fold=None
            )
        )

        new_model = BaseModel.load_from_checkpoint(
            trained_fusion_model_checkpoint,
            model=model.model.__class__(
                pred_type=params[
                    "pred_type"
                ],  # pred_type is a string (binary, regression, multiclass)
                data_dims=dm.data_dims,  # data_dims is a list of tuples
                params=params,  # params is a dict))
            ),
        )

        new_model.eval()
        # get the predictions
        end_outputs_list = []
        logits_list = []
        reals_list = []

        for batch in dataloader:
            x, y = new_model.get_data_from_batch(batch)
            out = new_model.get_model_outputs_and_loss(x, y)
            loss, end_output, logits = out

            end_outputs_list.append(end_output.cpu().detach())
            logits_list.append(logits.cpu().detach())
            reals_list.append(y.cpu().detach())

        # get the train reals, train preds, val reals, val preds
        train_reals = model.train_reals.cpu()
        train_preds = model.train_preds.cpu()
        val_preds = torch.cat(end_outputs_list, dim=-1)
        val_reals = torch.cat(reals_list, dim=-1)
        val_logits = torch.cat(logits_list, dim=0)

        # metrics
        metric_values = {}

        for metric in new_model.metrics[
            new_model.model.pred_type
        ]:  # loop through the metrics
            if "auroc" in metric["name"]:
                predicted = val_logits  # AUROC needs logits
            else:
                predicted = val_preds

            metric_val = metric["metric"](
                new_model.safe_squeeze(predicted),
                new_model.safe_squeeze(val_reals),
            )

            metric_values[metric["name"]] = metric_val

        return train_reals, train_preds, val_reals, val_preds, metric_values


class RealsVsPreds(ParentPlotter):
    """
    Plots the real values vs the predicted values for a model.
    Pink dots are the training data, green dots are the validation data. The validation data is
    either new data if using from_new_data or the original validation data if using from_final_val_data.

    Parameters
    ----------
    which_data: str
        Either "from_new_data" or "from_final_val_data".
    model: nn.Module
        The trained model.
    X: torch.Tensor
        The data to plot.
    y: torch.Tensor
        The labels for the data.
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def from_new_data(
        self, model, params, data_file_suffix="_test", checkpoint_file_suffix=None
    ):
        """

        Parameters
        ----------
        data_file_suffix: str
            Suffix for the data file e.g. _test means that the source files are
            ``params["tabular1_source_test]``, ``params["tabular2_source_test]``,
            ``params["img_source_test]``.
            Change this to whatever suffix you have chosen for your new data files.
        """

        if not isinstance(model, list):
            raise ValueError(
                (
                    "Argument 'model' is not a list. "
                    "Please check the model and the function input."
                    "If you are using a train/test model, the single model must be in a list of length 1."
                )
            )

        if len(model) > 1:
            # if isinstance(model, list):  # kfold model
            if not model[0].model.params["kfold_flag"]:
                raise ValueError(
                    (
                        "Argument 'model' is a list but kfold_flag is False. "
                        "Please check the model and the function input."
                    )
                )

            model_list = model

            (
                train_reals,
                train_preds,
                val_reals,
                val_preds,
                metrics_per_fold,
                overall_kfold_metrics,
            ) = self.get_new_kfold_data(
                model_list, params, data_file_suffix, checkpoint_file_suffix
            )

            figure = self.reals_vs_preds_kfold(
                model_list,
                val_reals,
                val_preds,
                metrics_per_fold,
                overall_kfold_metrics,
            )

            figure.suptitle("From new data")

        elif len(model) == 1:
            # isinstance(model, nn.Module):  # train/test model
            if model[0].model.params["kfold_flag"]:
                raise ValueError(
                    (
                        "Argument 'model' is a list of one model but kfold_flag is True. "
                        "Please check the model and the function input (k must be larger than 1)."
                    )
                )

            (
                train_reals,
                train_preds,
                val_reals,
                val_preds,
                metric_values,
            ) = self.get_new_tt_data(
                model, params, data_file_suffix, checkpoint_file_suffix
            )

            # plot the figure
            figure = self.reals_vs_preds_tt(
                model, train_reals, train_preds, val_reals, val_preds, metric_values
            )

            figure.suptitle("From new data")

        else:
            raise ValueError(("Argument 'model' is an empty list. "))

        return figure

    @classmethod
    def from_final_val_data(self, model):
        if not isinstance(model, list):
            raise ValueError(
                (
                    "Argument 'model' is not a list. "
                    "Please check the model and the function input."
                    "If you are using a train/test model, the single model must be in a list of length 1."
                )
            )

        if len(model) > 1:  # kfold model (list of models and their checkpoints)
            # if isinstance(model[0], list):  # kfold model
            if not model[0].model.params["kfold_flag"]:
                raise ValueError(
                    (
                        "Argument 'model' is a list of length > 1 but kfold_flag is False. "
                        "Please check the model and the function input."
                    )
                )

            model_list = (
                model  # renaming for clarity that this is a list of trained models
            )

            (
                train_reals,
                train_preds,
                val_reals,
                val_preds,
                metrics_per_fold,
                overall_kfold_metrics,
            ) = self.get_kfold_data_from_model(model_list)

            figure = self.reals_vs_preds_kfold(
                model_list,
                val_reals,
                val_preds,
                metrics_per_fold,
                overall_kfold_metrics,
            )

            figure.suptitle("From final val data")

        elif len(model) == 1:
            # isinstance(model[0], nn.Module):  # train/test model

            if model[0].model.params["kfold_flag"]:
                raise ValueError(
                    (
                        "Argument 'model' is a list of one model+checkpoint but kfold_flag is True. "
                        "Please check the model and the function input."
                    )
                )
            # get the data
            (
                train_reals,
                train_preds,
                val_reals,
                val_preds,
                metric_values,
            ) = self.get_tt_data_from_model(model)

            # plot the figure
            figure = self.reals_vs_preds_tt(
                model, train_reals, train_preds, val_reals, val_preds, metric_values
            )

            figure.suptitle("From final val data")

        else:
            raise ValueError(("Argument 'model' is an empty list. "))

        return figure

    @classmethod
    def reals_vs_preds_kfold(
        self,
        model_list,
        val_reals,
        val_preds,
        metrics_per_fold,
        overall_kfold_metrics,
    ):
        first_fold_model = model_list[0]
        metric_names = list(metrics_per_fold.keys())
        N = first_fold_model.model.params["num_k"]

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
                f"Fold {n+1}: {metric_names[0]}={float(metrics_per_fold[metric_names[0]][n]):.3f}"
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
        self, model, train_reals, train_preds, val_reals, val_preds, metric_values
    ):
        # plot for train/test reals v preds
        # called from from_new_data or from_final_val_data
        # takes in data from either from_new_data or from_final_val_data
        # returns a list of figures or dict of figures

        model = model[0]

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
    def __init__(self):
        super().__init__()

    @classmethod
    def from_new_data(self, model, params, data_file_suffix="_test"):
        """

        Parameters
        ----------
        data_file_suffix: str
            Suffix for the data file e.g. _test means that the source files are
            ``params["tabular1_source_test]``, ``params["tabular2_source_test]``,
            ``params["img_source_test]``.
            Change this to whatever suffix you have chosen for your new data files.
        """

        if not isinstance(model, list):
            raise ValueError(
                (
                    "Argument 'model' is not a list. "
                    "Please check the model and the function input."
                    "If you are using a train/test model, the single model must be in a list of length 1."
                )
            )

        if len(model) > 1:  # kfold model
            if not model[0].model.params["kfold_flag"]:
                raise ValueError(
                    (
                        "Argument 'model' is a list but kfold_flag is False. "
                        "Please check the model and the function input."
                    )
                )

            model_list = model

            (
                train_reals,
                train_preds,
                val_reals,
                val_preds,
                metrics_per_fold,
                overall_kfold_metrics,
            ) = self.get_new_kfold_data(model_list, params, data_file_suffix)

            figure = self.confusion_matrix_kfold(
                model_list,
                val_reals,
                val_preds,
                metrics_per_fold,
                overall_kfold_metrics,
            )

        elif len(model) == 1:  # train/test model
            if model[0].model.params["kfold_flag"]:
                raise ValueError(
                    (
                        "Argument 'model' is a list of one model but kfold_flag is True. "
                        "Please check the model and the function input (k must be larger than 1)."
                    )
                )

            (
                train_reals,
                train_preds,
                val_reals,
                val_preds,
                metric_values,
            ) = self.get_new_tt_data(model, params, data_file_suffix)

            # plot the figure
            figure = self.confusion_matrix_tt(val_reals, val_preds, metric_values)

        else:
            raise ValueError(("Argument 'model' is an empty list. "))

        return figure

    @classmethod
    def from_final_val_data(self, model):
        if not isinstance(model, list):
            raise ValueError(
                (
                    "Argument 'model' is not a list. "
                    "Please check the model and the function input."
                    "If you are using a train/test model, the single model must be in a list of length 1."
                )
            )

        if len(model) > 1:  # kfold model
            if not model[0].model.params["kfold_flag"]:
                raise ValueError(
                    (
                        "Argument 'model' is a list but kfold_flag is False. "
                        "Please check the model and the function input."
                    )
                )

            model_list = (
                model  # renaming for clarity that this is a list of trained models
            )

            (
                train_reals,
                train_preds,
                val_reals,
                val_preds,
                metrics_per_fold,
                overall_kfold_metrics,
            ) = self.get_kfold_data_from_model(model_list)

            figure = self.confusion_matrix_kfold(
                model_list,
                val_reals,
                val_preds,
                metrics_per_fold,
                overall_kfold_metrics,
            )

        elif len(model) == 1:  # train/test model
            if model[0].model.params["kfold_flag"]:
                raise ValueError(
                    (
                        "Argument 'model' is a list of one model but kfold_flag is True. "
                        "Please check the model and the function input (k must be larger than 1)."
                    )
                )

            self.model = model

            (
                train_reals,
                train_preds,
                val_reals,
                val_preds,
                metric_values,
            ) = self.get_tt_data_from_model(model)

            figure = self.confusion_matrix_tt(val_reals, val_preds, metric_values)

        else:
            raise ValueError(("Argument 'model' is an empty list. "))

        return figure

    @classmethod
    def confusion_matrix_tt(self, val_reals, val_preds, metric_values):
        # plot for kfold confusion matrix

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
            f"{self.model[0].model.method_name} - Validation {metric1_name}: {float(metric_values[metric1_name]):.3f}"
        )

        plt.tight_layout()

        return fig

    @classmethod
    def confusion_matrix_kfold(
        self,
        model_list,
        val_reals,
        val_preds,
        metrics_per_fold,
        overall_kfold_metrics,
    ):
        first_fold_model = model_list[0]
        metric_names = list(metrics_per_fold.keys())
        N = first_fold_model.model.params["num_k"]

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
                f"Fold {n+1}:\n{metric_names[0]}={float(metrics_per_fold[metric_names[0]][n]):.3f}"
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
    """Plotting class for comparing models."""

    def __init__(self):
        super().__init__()

    @classmethod
    def from_final_val_data(self, model_list, kfold_flag):
        """
        Parameters
        ----------

        model_list: dict
            Dictionary of models. Keys are the model names and values are [[model, ckpt_path]*num_k]
        kfold_flag: bool
            Whether the models are kfold models or not.
        """
        comparing_models_metrics = {}

        if kfold_flag:
            overall_kfold_metrics_dict = {}  #
            for model in model_list:
                model = model_list[model]

                (
                    train_reals,
                    train_preds,
                    val_reals,
                    val_preds,
                    metrics_per_fold,
                    overall_kfold_metrics,
                ) = self.get_kfold_data_from_model(model)

                comparing_models_metrics[model[0].model.method_name] = metrics_per_fold

                overall_kfold_metrics_dict[
                    model[0].model.method_name
                ] = overall_kfold_metrics

            figure = self.kfold_comparison_plot(comparing_models_metrics)

            df = self.get_performance_dataframe(
                comparing_models_metrics, overall_kfold_metrics_dict, kfold_flag
            )

        else:
            for model in model_list:
                model = model_list[model]
                (
                    train_reals,
                    train_preds,
                    val_reals,
                    val_preds,
                    metric_values,
                ) = self.get_tt_data_from_model(model)

                comparing_models_metrics[model[0].model.method_name] = metric_values

            print("comparing models metrics", comparing_models_metrics)

            figure = self.train_test_comparison_plot(comparing_models_metrics)
            df = self.get_performance_dataframe(
                comparing_models_metrics, None, kfold_flag
            )

        return figure, df

    @classmethod
    def from_new_data(self, model_list):
        raise NotImplementedError

    @classmethod
    def kfold_comparison_plot(self, comparing_models_metrics):
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
        print("comp models metrics", comparing_models_metrics)
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
    def train_test_comparison_plot(self, comparing_models_metrics):
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
            y_indices,
            #   - bar_width / 2,
            metric_1_values,
            bar_width,
            color="violet",
            edgecolor="purple"
            # label=self.metric1name,
        )
        # ax[0].bar_label(bars1, fmt="%.2f", label_type="edge")

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
            #   + bar_width / 2,
            metric_2_values,
            bar_width,
            color="powderblue",
            edgecolor="steelblue",
            # label=self.metric2name,
        )
        # ax[1].bar_label(bars2, fmt="%.2f", label_type="edge")

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
        self, comparing_models_metrics, overall_kfold_metrics_dict, kfold_flag
    ):
        method_names = list(
            comparing_models_metrics.keys()
        )  # [method1name, method2name,...]
        metric1name = list(comparing_models_metrics[method_names[0]].keys())[0]
        metric2name = list(comparing_models_metrics[method_names[0]].keys())[1]

        if kfold_flag:
            # copy self.overall_kfold_metrics to a new dictionary
            # so that we can change the values from lists to single numbers

            overall_kfold_metrics_copy = overall_kfold_metrics_dict.copy()

            for method, metrics in overall_kfold_metrics_copy.items():
                for metric, value in metrics.items():
                    overall_kfold_metrics_copy[method][metric] = value.item()

            df = pd.DataFrame(overall_kfold_metrics_copy).transpose()

            # Create a DataFrame for overall kfold metrics
            folds_df = pd.DataFrame(comparing_models_metrics).T.reset_index()
            folds_df.rename(columns={"index": "Method"}, inplace=True)

            num_folds = len(folds_df[metric1name][0])
            fold_columns_metric1 = [
                f"fold{i+1}_{metric1name}" for i in range(num_folds)
            ]
            fold_columns_metric2 = [
                f"fold{i+1}_{metric2name}" for i in range(num_folds)
            ]

            for i, col in enumerate(fold_columns_metric1):
                folds_df[fold_columns_metric1[i]] = folds_df[metric1name].apply(
                    lambda x: x[i] if len(x) > i else None
                )
                folds_df[fold_columns_metric2[i]] = folds_df[metric2name].apply(
                    lambda x: x[i] if len(x) > i else None
                )

            folds_df.drop(columns=[metric1name, metric2name], inplace=True)
            folds_df.set_index("Method", inplace=True)

            final_df = pd.concat([df, folds_df], axis=1)

            return final_df

        else:
            # Reshape the data into a list of dictionaries
            reshaped_data = []
            for method, metrics in comparing_models_metrics.items():
                reshaped_data.append({"Method": method, **metrics})

            # Create a DataFrame from the reshaped data
            df = pd.DataFrame(reshaped_data)
            df.set_index("Method", inplace=True)
            df.index.name = None

            return df
