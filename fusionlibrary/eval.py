"""
Functions for evaluating the performance of the models and plotting the results.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import math
from matplotlib import gridspec
import torch_geometric as pyg
import networkx as nx
from sklearn.manifold import TSNE
import pandas as pd
import torch.nn as nn


class Plotter:
    """
    Class for plotting the results of the models.

    Attributes
    ----------
    trained_model_dict : dict
        Dictionary of trained models.
    model_names : list
        List of model names.
    params : dict
        Dictionary of parameters.
    pred_type : str
        Prediction type.
    metric1name : str
        Name of metric 1.
    metric1func : function
        Function for calculating metric 1.
    metric2name : str
        Name of metric 2.
    metric2func : function
        Function for calculating metric 2.

    """

    def __init__(self, trained_model_dict, params):
        """
        Initialize the Plotter instance.

        Parameters
        ----------
        trained_model_dict : dict
            Dictionary of trained models.
        params : dict
            Dictionary of parameters.
        """

        super().__init__()

        # Initialise variables for use later in the class
        self.trained_model_dict = trained_model_dict
        self.model_names = list(trained_model_dict.keys())
        print("Plotting models", self.model_names, "...")

        self.params = params
        self.pred_type = self.params["pred_type"]  # for less verbose access

        # getting first model to get metric names and functions for less verbose access
        if params["kfold_flag"]:
            first_model = self.trained_model_dict[self.model_names[0]][0]
        else:
            first_model = self.trained_model_dict[self.model_names[0]]

        # metric names for less verbose access (they are consistent across models)
        self.metric1name = first_model.metrics[self.pred_type][0]["name"]
        self.metric1func = first_model.metrics[self.pred_type][0]["metric"]

        self.metric2name = first_model.metrics[self.pred_type][1]["name"]
        self.metric2func = first_model.metrics[self.pred_type][1]["metric"]

        # if there is one model
        if len(self.model_names) == 1:
            print("Plotting results of a single model.")
        else:
            print(
                "Plotting comparison plots for",
                len(self.model_names),
                "models:",
                self.model_names,
            )

        self.single_model_tt_plots = {
            "regression": self.reals_vs_preds,
            "binary": self.confusion_matrix_plotter,
            "multiclass": self.confusion_matrix_plotter,
        }

        self.single_model_kfold_plots = {
            "regression": self.reals_vs_preds_kfold,
            "binary": self.confusion_matrix_plotter_kfold,
            "multiclass": self.confusion_matrix_plotter_kfold,
        }

        self.multi_model_tt_plots = {
            "regression": self.compare_bar_chart,  # csv saving? bar chart?
            "binary": self.compare_bar_chart,
            "multiclass": self.compare_bar_chart,
        }

        self.multi_model_kfold_plots = {
            "regression": self.compare_violin_plot,  # violin plots of the folds
            "binary": self.compare_violin_plot,
            "multiclass": self.compare_violin_plot,
        }

    def get_kfold_numbers(self):
        """
        Gets the lists of train_reals, train_preds, val_reals, val_preds for all folds and saves to self.
        Gets the altogether list of val_reals and val_preds over folds and saves to self.
        Gets metrics to put in the plot titles, adds to metrics list.

        Returns
        -------
        None
        """

        # create empty lists for the folds of train_reals, train_preds, val_reals, val_preds
        self.train_reals = []
        self.train_preds = []
        self.val_reals = []
        self.val_preds = []
        self.val_logits = []

        # dictionary to store the metrics for each fold
        self.kfold_plot_val_accs = {self.metric1name: [], self.metric2name: []}

        self.trained_model_list = self.trained_model_dict[self.current_model_name]

        for k in range(self.params["num_k"]):
            k_trained_model_var = vars(self.trained_model_list[k])

            self.trained_model_list[k].eval()

            self.train_reals.append(k_trained_model_var["train_reals"].cpu())
            self.train_preds.append(k_trained_model_var["train_preds"].cpu())
            self.val_reals.append(k_trained_model_var["val_reals"].cpu())
            self.val_preds.append(k_trained_model_var["val_preds"].cpu())
            self.val_logits.append(k_trained_model_var["val_logits"].cpu())

            self.kfold_plot_val_accs[self.metric1name].append(
                k_trained_model_var["metric1"]
            )
            self.kfold_plot_val_accs[self.metric2name].append(
                k_trained_model_var["metric2"]
            )

        # altogether validation sets: concatenate all the lists of val_reals and val_preds
        self.all_preds = torch.cat(self.val_preds, dim=-1)
        self.all_reals = torch.cat(self.val_reals, dim=-1)
        self.all_logits = torch.cat(self.val_logits, dim=0)

        self.plot_val_accs = {}

        # saving the metrics for the altogether validation sets
        for i, metric in enumerate(self.trained_model_list[0].metrics[self.pred_type]):
            if "auroc" in metric["name"]:
                predicted = self.all_logits  # AUROC needs logits
            else:
                predicted = self.all_preds

            val_step_acc = metric["metric"](
                self.trained_model_list[0].safe_squeeze(predicted),
                self.trained_model_list[0].safe_squeeze(self.all_reals),
            )
            self.plot_val_accs[metric["name"]] = val_step_acc

    def get_train_test_numbers(self):
        """
        Gets the lists of train_reals, train_preds, val_reals, val_preds and saves to self.
        Gets metrics to put in the plot titles, adds to metrics list.

        Returns
        -------
        None
        """

        self.trained_model = self.trained_model_dict[self.current_model_name]
        self.trained_model.eval()

        # get lists of train_reals, train_preds, val_reals, val_preds and save to self
        self.train_reals = vars(self.trained_model_dict[self.current_model_name])[
            "train_reals"
        ].cpu()
        self.train_preds = vars(self.trained_model_dict[self.current_model_name])[
            "train_preds"
        ].cpu()
        self.val_reals = vars(self.trained_model_dict[self.current_model_name])[
            "val_reals"
        ].cpu()
        self.val_preds = vars(self.trained_model_dict[self.current_model_name])[
            "val_preds"
        ].cpu()

        self.plot_val_accs = {
            self.metric1name: vars(self.trained_model_dict[self.current_model_name])[
                "metric1"
            ],
            self.metric2name: vars(self.trained_model_dict[self.current_model_name])[
                "metric2"
            ],
        }

    def plot_all(self):
        """
        Plots all the results.
        """

        # if there is only one model
        if len(self.model_names) == 1:
            self.current_model_name = self.model_names[0]

            if self.params["kfold_flag"]:
                # get lists of train_reals, train_preds, val_reals, val_preds for all folds and save to self

                self.get_kfold_numbers()

                results_figs_dict = self.single_model_kfold_plots[self.pred_type]()

                return results_figs_dict

            else:
                self.get_train_test_numbers()

                self.trained_model = self.trained_model_dict[self.model_names[0]]

                # plot the results for one train/test model
                results_figs_dict = self.single_model_tt_plots[self.pred_type]()

                return results_figs_dict

        # if there are multiple models, plot the comparison plots
        else:
            figures_dict = (
                {}
            )  # keys: figure names (including model name), values: figures

            if self.params["kfold_flag"]:
                self.comparing_models_metrics = (
                    {}
                )  # keys: model names, values: lists of kfold metrics len(k)

                self.overall_kfold_metrics = {}
                # or maybe value is another dictionary {"R2": [...,...,...], "MAE": [...,..,...]}

                for model_name in self.model_names:
                    self.current_model_name = model_name

                    self.get_kfold_numbers()
                    # get lists of train_reals, train_preds, val_reals, val_preds for all folds and save to self
                    # get altogether list of val_reals and val_preds over folds and save to self
                    # get metrics to put in the plot titles, add to metrics list
                    self.comparing_models_metrics[model_name] = self.kfold_plot_val_accs
                    self.overall_kfold_metrics[model_name] = self.plot_val_accs

                # plot comparison of all kfold models: violin plot?
                comparison_figs_dict = self.multi_model_kfold_plots[self.pred_type]()
                figures_dict.update(comparison_figs_dict)

                # self.save_performance_csv()
                # incorporate self.plot_val_accs (overall kfold performance) into this csv

                return figures_dict

            else:
                self.comparing_models_metrics = (
                    {}
                )  # keys: model names, values: metric values (single numbers)
                # or maybe value is another dictionary {"R2": ... "MAE": ...}
                for model_name in self.model_names:
                    # get train_reals, train_preds, val_reals, val_preds and save to self
                    self.current_model_name = model_name
                    self.get_train_test_numbers()

                    # plot the predefined models for self.params["pred_type"]
                    self.comparing_models_metrics[model_name] = self.plot_val_accs

                # plot comparison of all train/test models: bar chart?
                comparison_figs_dict = self.multi_model_tt_plots[self.pred_type]()
                figures_dict.update(comparison_figs_dict)

                # self.save_performance_csv()

                return figures_dict

    def get_performance_df(self):
        """
        Saves the performance metrics to a CSV file.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the performance metrics.
        """
        if self.params["kfold_flag"]:
            # copy self.overall_kfold_metrics to a new dictionary
            # so that we can change the values from lists to single numbers

            overall_kfold_metrics_copy = self.overall_kfold_metrics.copy()

            for method, metrics in overall_kfold_metrics_copy.items():
                for metric, value in metrics.items():
                    overall_kfold_metrics_copy[method][metric] = value.item()

            df = pd.DataFrame(overall_kfold_metrics_copy).transpose()

            # Create a DataFrame for overall kfold metrics
            folds_df = pd.DataFrame(self.comparing_models_metrics).T.reset_index()
            folds_df.rename(columns={"index": "Method"}, inplace=True)

            # num_folds = len(folds_df[self.metric1name][0])
            fold_columns_metric1 = [
                f"fold{i+1}_{self.metric1name}" for i in range(self.params["num_k"])
            ]
            fold_columns_metric2 = [
                f"fold{i+1}_{self.metric2name}" for i in range(self.params["num_k"])
            ]

            for i, col in enumerate(fold_columns_metric1):
                folds_df[fold_columns_metric1[i]] = folds_df[self.metric1name].apply(
                    lambda x: x[i] if len(x) > i else None
                )
                folds_df[fold_columns_metric2[i]] = folds_df[self.metric2name].apply(
                    lambda x: x[i] if len(x) > i else None
                )

            folds_df.drop(columns=[self.metric1name, self.metric2name], inplace=True)
            folds_df.set_index("Method", inplace=True)

            final_df = pd.concat([df, folds_df], axis=1)

            return final_df

        else:
            # Reshape the data into a list of dictionaries
            reshaped_data = []
            for method, metrics in self.comparing_models_metrics.items():
                reshaped_data.append({"Method": method, **metrics})

            # Create a DataFrame from the reshaped data
            df = pd.DataFrame(reshaped_data)
            df.set_index("Method", inplace=True)
            df.index.name = None

            return df

    def reals_vs_preds(self):
        """
        Plots the real values against the predicted values for the training and validation sets.
        """

        fig, ax = plt.subplots()

        ax.scatter(
            self.train_reals,
            self.train_preds,
            c="#f082ef",
            marker="o",
            label="Train",
        )
        ax.scatter(
            self.val_reals, self.val_preds, c="#00b64e", marker="^", label="Validation"
        )

        # Get the limits of the current scatter plot
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()

        # Set up data points for the x=y line
        line_x = np.linspace(min(x_min, y_min), max(x_max, y_max), 100)
        line_y = line_x

        # Plot the x=y line as a dashed line
        plt.plot(line_x, line_y, linestyle="dashed", color="black", label="x=y Line")

        ax.set_title(
            f"{self.current_model_name} - Validation {self.metric1name}: {float(self.plot_val_accs[self.metric1name]):.3f}"
        )

        ax.set_xlabel("Real Values")
        ax.set_ylabel("Predictions")
        ax.legend()

        return {f"{self.current_model_name}_reals_vs_preds": fig}

    def confusion_matrix_plotter(self):
        """
        Plots the confusion matrix of a train/test model.
        """
        # Get the confusion matrix for the validation set

        conf_matrix = confusion_matrix(y_true=self.val_reals, y_pred=self.val_preds)

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

        plt.title(
            f"Validation {self.metric1name}: {float(self.plot_val_accs[self.metric1name]):.3f}"
        )

        plt.tight_layout()

        return {f"{self.current_model_name}_confusion_matrix": fig}

    def reals_vs_preds_kfold(self):
        """
        Plots the real values against the predicted values for the training and validation sets.
        For kfold models.
        """

        # concatenate real and pred values from all folds

        N = self.params["num_k"]
        cols = 3
        rows = int(math.ceil(N / cols))

        gs = gridspec.GridSpec(rows, cols)
        reals_vs_preds_fig = plt.figure()
        for n in range(N):
            if n == 0:
                ax = reals_vs_preds_fig.add_subplot(gs[n])
                ax_og = ax
            else:
                ax = reals_vs_preds_fig.add_subplot(gs[n], sharey=ax_og, sharex=ax_og)

            # get real and predicted values for the current fold
            reals = self.val_reals[n]
            preds = self.val_preds[n]

            # plot real vs. predicted values
            ax.scatter(reals, preds, c="#f082ef", marker="o")

            # plot x=y line as a dashed line
            ax.plot(
                [0, 1],
                [0, 1],
                color="k",
                linestyle="--",
                alpha=0.75,
                zorder=0,
                transform=ax.transAxes,
            )

            # set title of plot to the metric for the current fold
            ax.set_title(
                f"Fold {n+1}: R2={float(self.kfold_plot_val_accs[self.metric1name][n]):.3f}"
            )

        plt.suptitle(f"{self.current_model_name}: reals vs. predicteds")

        reals_vs_preds_fig.tight_layout()

        # plot all real vs. predicted values
        together_reals_v_preds_fig, ax1 = plt.subplots()
        ax1.scatter(self.all_reals, self.all_preds, c="#f082ef", marker="o")

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
        ax1.set_title(
            f"{self.current_model_name}: {self.metric1name}={float(self.plot_val_accs[self.metric1name]):.3f}"
        )
        together_reals_v_preds_fig.tight_layout()

        # TODO combine these images to be pretty without worrying about what k is? ask chatgpt
        # return [reals_vs_preds_fig, together_reals_v_preds_fig]
        return {
            f"{self.current_model_name}_reals_vs_preds_kfold": reals_vs_preds_fig,
            f"{self.current_model_name}_reals_vs_preds_kfold_together": together_reals_v_preds_fig,
        }

    def confusion_matrix_plotter_kfold(self):
        """
        Plots the confusion matrix of a kfold model.
        """

        N = self.params["num_k"]
        cols = 3
        rows = int(math.ceil(N / cols))

        gs = gridspec.GridSpec(rows, cols)
        k_fold_confusion_matrix_fig = plt.figure()
        for n in range(N):
            if n == 0:
                ax = k_fold_confusion_matrix_fig.add_subplot(gs[n])
                ax_og = ax
            else:
                ax = k_fold_confusion_matrix_fig.add_subplot(
                    gs[n], sharey=ax_og, sharex=ax_og
                )

            # get real and predicted values for the current fold
            reals = self.val_reals[n]
            preds = self.val_preds[n]

            # confusion plot time
            # Get the confusion matrix for the validation set
            conf_matrix = confusion_matrix(y_true=reals, y_pred=preds.squeeze())

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
                        size="large",
                    )

            ax.set_xlabel("Predictions", fontsize=10)
            ax.set_ylabel("Actuals", fontsize=10)

            ax.set_title(
                f"Fold {n+1} {self.metric1name}: \n{float(self.kfold_plot_val_accs[self.metric1name][n]):.3f}"
            )

        plt.suptitle(f"{self.current_model_name}: confusion matrices")
        k_fold_confusion_matrix_fig.tight_layout()

        # altogether confusion matrix
        together_k_fold_confusion_matrix_fig, ax1 = plt.subplots()
        # Get the confusion matrix for the validation set
        conf_matrix = confusion_matrix(
            y_true=self.all_reals, y_pred=self.all_preds.squeeze()
        )

        # Plot the confusion matrix as a heatmap
        ax1.matshow(conf_matrix, cmap=plt.cm.RdPu, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                # Add the value of each cell to the plot
                ax1.text(
                    x=j,
                    y=i,
                    s=conf_matrix[i, j],
                    va="center",
                    ha="center",
                    size="xx-large",
                )

        ax1.set_xlabel("Predictions", fontsize=18)
        ax1.set_ylabel("Actuals", fontsize=18)
        ax1.set_title(
            f"{self.current_model_name}: {self.metric1name} = {float(self.plot_val_accs[self.metric1name]):.3f}"
        )
        together_k_fold_confusion_matrix_fig.tight_layout()

        return {
            f"{self.current_model_name}_confusion_matrix_kfold": k_fold_confusion_matrix_fig,
            f"{self.current_model_name}_confusion_matrix_kfold_together": together_k_fold_confusion_matrix_fig,
        }

    def compare_violin_plot(self):
        """
        Plots a violin plot comparing the results of multiple kfold models.
        """

        # get method names and metric names
        method_names = list(
            self.comparing_models_metrics.keys()
        )  # [method1name, method2name,...]

        # get metric values for each method
        metric_1_values = [
            self.comparing_models_metrics[method][self.metric1name]
            for method in method_names
        ]
        metric_2_values = [
            self.comparing_models_metrics[method][self.metric2name]
            for method in method_names
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
        ax[0].set_title(self.metric1name)
        ax[1].set_title(self.metric2name)
        ax[1].set_xlim(left=0.0)

        plt.suptitle("Distribution of metrics between cross-validation folds")

        plt.tight_layout()

        return {"compare_kfold_models": fig}

    def compare_bar_chart(self):
        """
        Plots a bar chart comparing the results of multiple train/test models.
        """

        # get method names and metric names
        method_names = list(
            self.comparing_models_metrics.keys()
        )  # [method1name, method2name,...]

        # get metric values for each method
        metric_1_values = [
            self.comparing_models_metrics[method][self.metric1name]
            for method in method_names
        ]
        metric_2_values = [
            self.comparing_models_metrics[method][self.metric2name]
            for method in method_names
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
        ax[0].set_title(self.metric1name)
        ax[1].set_title(self.metric2name)
        ax[1].set_xlim(left=0.0)

        # Show the plot
        plt.suptitle("Model Performance Comparison")
        plt.tight_layout()

        return {"compare_tt_models": fig}

    def save_to_local(self, plots_dict, extra_string=""):
        """
        Save dictionary of plots to local directory.
        """

        for figure_name, figure in plots_dict.items():
            figure.savefig(
                f"{self.params['local_fig_path']}/{figure_name}{extra_string}.png"
            )
            plt.close(figure)

    def show_all(self, plots_dict):
        """
        Show all plots in dictionary.
        """
        for figure_name, figure in plots_dict.items():
            print(figure_name)
            figure.show()


#####################################
# Graph plotting functions (not used)
#####################################


# def plot_graph(graph_data, params):
#     """
#     Plots the graph structure.

#     Parameters
#     ----------
#     graph_data: pyg.data.Data
#         Graph data.
#     params: dict
#         Additional parameters.
#     """

#     # convert graph data to networkx graph
#     G = pyg.utils.convert.to_networkx(
#         graph_data,
#         to_undirected=True,
#         remove_self_loops=True,
#         node_attrs=["x"],
#         edge_attrs=["edge_attr"],
#     )
#     for u, v, d in G.edges(data=True):
#         d["weight"] = d["edge_attr"]
#         # d['nodecolor'] = d["y"].item()

#     # colour nodes with their label
#     for n, d in G.nodes(data=True):
#         d["nodecolor"] = graph_data.y[n].item()

#     edges, weights = zip(*nx.get_edge_attributes(G, "weight").items())
#     nodes, colors = zip(*nx.get_node_attributes(G, "nodecolor").items())

#     pos = nx.spectral_layout(G)

#     # draw graph with node colour and edge weight
#     nx.draw(
#         G,
#         pos,
#         node_color=colors,
#         edgelist=edges,
#         edge_color=weights,
#         width=5.0,
#         edge_cmap=plt.cm.Blues,
#         cmap=plt.cm.coolwarm,
#         node_size=50,
#     )
#     vmin = np.min(weights)
#     vmax = np.max(weights)
#     sm = plt.cm.ScalarMappable(
#         cmap=plt.cm.Blues, norm=plt.Normalize(vmin=vmin, vmax=vmax)
#     )
#     sm.set_array([])
#     cbar = plt.colorbar(sm)

#     if params["cluster"] is False and params["log"] is False:
#         # save graph structure plot as a png locally
#         plt.savefig(f"{params['local_fig_path']}/graph_structure", dpi=180)
#         plt.close()


# def visualise_graphspace(h, color, params, path_suffix, method_name):
#     """
#     Visualizes the graph space using t-SNE.

#     Parameters
#     ----------
#     h: torch.Tensor
#         Graph data.
#     color: list
#         List of colors.
#     params: dict
#         Additional parameters.
#     path_suffix: str
#         Suffix for the saved image.
#     method_name: str
#         Name of the method.

#     Returns
#     -------
#     fig: matplotlib.figure.Figure
#         The figure containing the t-SNE plot.
#     """

#     z = TSNE(n_components=2, perplexity=30).fit_transform(
#         h.reshape(-1, 1).detach().cpu().numpy()
#     )

#     fig, ax = plt.subplots(figsize=(10, 10))

#     ax.set_xticks([])
#     ax.set_yticks([])

#     scatter = ax.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
#     cbar = fig.colorbar(scatter)
#     cbar.set_label("Color Legend")
#     ax.set_title(f"Method: {method_name}")
#     return fig

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PLOTS FOR ONE MODEL~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class ParentPlotter:
    """Parent class for all plot classes."""

    def __init__(self):
        pass

    @classmethod
    def get_kfold_data_from_model(self, model_list):
        print("getting kfold data from model")

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
        for fold in model_list:
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
        print("getting train/test data from model")

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
    def get_new_kfold_data(self):
        print("getting new kfold data")

    @classmethod
    def get_new_tt_data(self):
        print("getting new train/test data")


class RealsVsPreds(ParentPlotter):
    # (which_data, model, X=None, y=None):
    """
    which_data: should be from_new_data or from_final_val_data
    model: fusion_model or list of models (for kfold)
    X: if from_new_data, then X is the new data
    y: if from_new_data, then y is the new data labels
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def from_new_data(self, model, X, y):
        # run X, y through the models and get the predictions
        # calculate metricS

        self.get_new_kfold_data()
        self.reals_vs_preds_kfold("calling reals_vs_preds_kfold from from_new_data")
        print("from_new_data")

    @classmethod
    def from_final_val_data(self, model):
        # get the predictions from the models (already saved in the model)
        # get the metrics from the models (already saved in the model)

        if isinstance(model, list):  # kfold model
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

            figure = self.reals_vs_preds_kfold(
                model_list,
                val_reals,
                val_preds,
                metrics_per_fold,
                overall_kfold_metrics,
            )

        elif isinstance(model, nn.Module):  # train/test model
            # self.model = model
            print("model name", model.model.__class__.__name__)
            print("pred_type", model.model.pred_type)

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

        else:
            raise ValueError(
                (
                    "Argument 'model' is not a list or nn.Module. "
                    "Please check the model and the function input."
                )
            )

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

        gs = gridspec.GridSpec(rows, cols)

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
                f"Fold {n+1}: R2={float(metrics_per_fold[metric_names[0]][n]):.3f}"
            )

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
    def from_new_data(self, model, X, y):
        # run X, y through the models and get the predictions
        # calculate metricS

        self.get_new_kfold_data()
        self.reals_vs_preds_kfold("calling reals_vs_preds_kfold from from_new_data")
        print("from_new_data")

    @classmethod
    def from_final_val_data(self, model):
        # get the predictions from the models (already saved in the model)
        # get the metrics from the models (already saved in the model)

        self.model = model
        print("model name", model.model.__class__.__name__)
        print("pred_type", model.model.pred_type)

        if model.model.params["kfold_flag"]:
            self.get_kfold_data_from_model()
            self.confusion_matrix_kfold(
                "calling confusion_matrix_kfold from from_final_val_data"
            )

        else:
            (
                train_reals,
                train_preds,
                val_reals,
                val_preds,
                metric_values,
            ) = self.get_tt_data_from_model()

            figure = self.confusion_matrix_tt(val_reals, val_preds, metric_values)

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
            f"{self.model.model.method_name} - Validation {metric1_name}: {float(metric_values[metric1_name]):.3f}"
        )

        plt.tight_layout()

        return fig

    @classmethod
    def confusion_matrix_kfold(self, val_reals, val_preds, metric_values):
        pass


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PLOTS FOR ONE MODEL~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
