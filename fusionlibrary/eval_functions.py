"""
Functions for evaluating the performance of the models and plotting the results.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import math
from matplotlib import gridspec
import wandb
import torch_geometric as pyg
import networkx as nx
from sklearn.manifold import TSNE
import pandas as pd


class Plotter:
    """
    Class for plotting the results of the models.

    Parameters
    ----------
    trained_model_dict : dict
        Dictionary of trained models.
    params : dict
        Dictionary of parameters.

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
        super().__init__()

        # initialising variables for use later in the class
        self.trained_model_dict = trained_model_dict
        self.model_names = list(trained_model_dict.keys())
        print("model names:", self.model_names)

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

            self.train_reals.append(k_trained_model_var["train_reals"])
            self.train_preds.append(k_trained_model_var["train_preds"])
            self.val_reals.append(k_trained_model_var["val_reals"])
            self.val_preds.append(k_trained_model_var["val_preds"])
            self.val_logits.append(k_trained_model_var["val_logits"])

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

        # # saving the metrics for the altogether validation sets
        # self.plot_val_accs = {
        #     self.metric1name: self.metric1func.to(self.trained_model_list[0].device)(
        #         self.all_preds, self.all_reals
        #     ),
        #     self.metric2name: self.metric2func.to(self.trained_model_list[0].device)(
        #         self.all_preds, self.all_reals
        #     ),
        # }

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
        ]
        self.train_preds = vars(self.trained_model_dict[self.current_model_name])[
            "train_preds"
        ]
        self.val_reals = vars(self.trained_model_dict[self.current_model_name])[
            "val_reals"
        ]
        self.val_preds = vars(self.trained_model_dict[self.current_model_name])[
            "val_preds"
        ]

        self.plot_val_accs = {
            self.metric1name: vars(self.trained_model_dict[self.current_model_name])[
                "metric1"
            ],
            self.metric2name: vars(self.trained_model_dict[self.current_model_name])[
                "metric2"
            ],
        }

        # self.plot_val_accs = {
        #     self.metric1name: self.metric1func.to(self.trained_model.device)(
        #         self.val_reals,
        #         self.val_preds,
        #     ).item(),
        #     self.metric2name: self.metric2func.to(self.trained_model.device)(
        #         self.val_reals,
        #         self.val_preds,
        #     ).item(),
        # }

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

                    # single_model_figs_dict = self.single_model_kfold_plots[
                    #     self.pred_type
                    # ]()
                    # figures_dict.update(single_model_figs_dict)

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

                    # results_figs_dict = self.single_model_tt_plots[self.pred_type]()
                    # figures_dict.update(results_figs_dict)

                # plot comparison of all train/test models: bar chart?
                comparison_figs_dict = self.multi_model_tt_plots[self.pred_type]()
                figures_dict.update(comparison_figs_dict)

                # self.save_performance_csv()

                return figures_dict

    def save_performance_csv(self):
        if self.params["kfold_flag"]:
            # copy self.overall_kfold_metrics to a new dictionary
            # so that we can change the values from lists to single numbers

            overall_kfold_metrics_copy = self.overall_kfold_metrics.copy()

            for method, metrics in overall_kfold_metrics_copy.items():
                for metric, value in metrics.items():
                    overall_kfold_metrics_copy[method][metric] = value.item()

            df = pd.DataFrame(overall_kfold_metrics_copy).transpose()
            # df.rename(
            #     columns={
            #         self.metric1name: f"val_{self.metric1name}",
            #         self.metric2name: f"overall_{self.metric2name}",
            #     },
            #     inplace=True,
            # )

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
            # models_df = pd.DataFrame(self.comparing_models_metrics)
            # models_df.to_csv("performance.csv")

    def reals_vs_preds(self):
        """
        Plots the real values against the predicted values for the training and validation sets.
        """

        fig, ax = plt.subplots()

        ax.scatter(self.train_reals, self.train_preds, c="blue", label="Train")
        ax.scatter(self.val_reals, self.val_preds, c="red", label="Validation")

        # Get the limits of the current scatter plot
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()

        # Set up data points for the x=y line
        line_x = np.linspace(min(x_min, y_min), max(x_max, y_max), 100)
        line_y = line_x

        # Plot the x=y line as a dashed line
        plt.plot(line_x, line_y, linestyle="dashed", color="black", label="x=y Line")

        ax.set_title(
            f"Validation {self.metric1name}: {float(self.plot_val_accs[self.metric1name]):.3f}"
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
        ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
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
            ax.scatter(reals, preds, marker="o")

            # plot x=y line as a dashed line
            ax.plot(
                [0, 1],
                [0, 1],
                color="r",
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
        ax1.scatter(self.all_reals, self.all_preds, marker="o")

        # plot x=y line as a dashed line
        ax1.plot(
            [0, 1],
            [0, 1],
            color="r",
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
            ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
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
        ax1.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
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

        fig, ax = plt.subplots()

        # Create the bar chart
        ax.barh(
            y_indices - bar_width / 2,
            metric_1_values,
            bar_width,
            label=self.metric1name,
        )
        ax.barh(
            y_indices + bar_width / 2,
            metric_2_values,
            bar_width,
            label=self.metric2name,
        )

        # Set x-axis labels and title
        ax.set_xlabel("Models")
        ax.set_ylabel("Scores")
        ax.set_title("Model Performance Comparison")
        ax.set_yticks(y_indices, method_names)
        ax.legend()

        # Show the plot
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

    def show_all(self, plots_dict):
        """
        Show all plots in dictionary.
        """
        for figure_name, figure in plots_dict.items():
            print(figure_name)
            figure.show()


# def reals_vs_preds(model, train_reals, train_preds, val_reals, val_preds):
#     """
#     Plots the real values against the predicted values for the training and validation sets.

#     Parameters
#     ----------

#     model: torch.nn.Module
#         The model to be evaluated.
#     train_reals: torch.Tensor
#         The real values for the training set.
#     train_preds: torch.Tensor
#         The predicted values for the training set.
#     val_reals: torch.Tensor
#         The real values for the validation set.
#     val_preds: torch.Tensor
#         The predicted values for the validation set.

#     Returns
#     -------

#     fig: matplotlib.figure.Figure
#         The figure containing the plot.
#     """
#     fig, ax = plt.subplots()

#     ax.scatter(train_reals, train_preds, c="blue", label="Train")
#     ax.scatter(val_reals, val_preds, c="red", label="Validation")
#     val_acc = model.metrics[model.pred_type][0]["metric"].to(model.device)(
#         val_preds, val_reals
#     )

#     # Get the limits of the current scatter plot
#     x_min, x_max = plt.xlim()
#     y_min, y_max = plt.ylim()

#     # Set up data points for the x=y line
#     line_x = np.linspace(min(x_min, y_min), max(x_max, y_max), 100)
#     line_y = line_x

#     # Plot the x=y line as a dashed line
#     plt.plot(line_x, line_y, linestyle="dashed", color="black", label="x=y Line")

#     ax.set_title(
#         f"Validation {model.metrics[model.pred_type][0]['name']}: {np.round(val_acc,3)}"
#     )

#     ax.set_xlabel("Real Values")
#     ax.set_ylabel("Predictions")
#     ax.legend()

#     return fig


# def confusion_matrix_plot(model, train_reals, train_preds, val_reals, val_preds):
#     """
#     Plots the confusion matrix.

#     Parameters
#     ----------
#     model: torch.nn.Module
#         The model to be evaluated.
#     train_reals: torch.Tensor
#         The real values for the training set.
#     train_preds: torch.Tensor
#         The predicted values for the training set.
#     val_reals: torch.Tensor
#         The real values for the validation set.
#     val_preds: torch.Tensor
#         The predicted values for the validation set.

#     Returns
#     -------
#     fig: matplotlib.figure.Figure
#         The figure containing the confusion matrix plot.
#     """

#     # Get the confusion matrix for the validation set
#     conf_matrix = confusion_matrix(y_true=val_reals, y_pred=val_preds)

#     # Create a figure and axis for the plot
#     fig, ax = plt.subplots(figsize=(7.5, 7.5))

#     # Plot the confusion matrix as a heatmap
#     ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
#     for i in range(conf_matrix.shape[0]):
#         for j in range(conf_matrix.shape[1]):
#             # Add the value of each cell to the plot
#             ax.text(
#                 x=j, y=i, s=conf_matrix[i, j], va="center", ha="center", size="xx-large"
#             )

#     plt.xlabel("Predictions", fontsize=18)
#     plt.ylabel("Actuals", fontsize=18)

#     # calculate the accuracy of the validation set using the model's metric
#     val_acc = model.metrics[model.pred_type][0]["metric"].to(model.device)(
#         val_preds, val_reals
#     )

#     plt.title(
#         f"Validation {model.metrics[model.pred_type][0]['name']}: {np.round(val_acc,3)}"
#     )

#     plt.tight_layout()

#     return fig


# def k_fold_reals_vs_preds(
#     reals_list,
#     preds_list,
#     overall_metric,  # overall metric for all folds
#     method_name,  # name of method
#     metric_list,  # list of metrics for each fold
#     params,
# ):
#     """
#     Plots real values against predicted values for each fold in k-fold cross-validation.

#     Parameters
#     ----------
#     reals_list: list of torch.Tensor
#         List of real values for each fold.
#     preds_list: list of torch.Tensor
#         List of predicted values for each fold.
#     overall_metric: float
#         Overall metric for all folds.
#     method_name: str
#         Name of the method.
#     metric_list: list of float
#         List of metrics for each fold.
#     params: dict
#         Additional parameters.

#     Returns
#     -------
#     fig_list: list of matplotlib.figure.Figure
#         List of figures containing the plots.
#     """

#     # concatenate real and pred values from all folds
#     all_preds = torch.cat(reals_list)
#     all_reals = torch.cat(preds_list)

#     N = params["num_k"]
#     cols = 3
#     rows = int(math.ceil(N / cols))

#     gs = gridspec.GridSpec(rows, cols)
#     reals_vs_preds_fig = plt.figure()
#     for n in range(N):
#         if n == 0:
#             ax = reals_vs_preds_fig.add_subplot(gs[n])
#             ax_og = ax
#         else:
#             ax = reals_vs_preds_fig.add_subplot(gs[n], sharey=ax_og, sharex=ax_og)

#         # get real and predicted values for the current fold
#         reals = reals_list[n]
#         preds = preds_list[n]

#         # plot real vs. predicted values
#         ax.scatter(reals, preds, marker="o")

#         # plot x=y line as a dashed line
#         ax.plot(
#             [0, 1],
#             [0, 1],
#             color="r",
#             linestyle="--",
#             alpha=0.75,
#             zorder=0,
#             transform=ax.transAxes,
#         )

#         # set title of plot to the metric for the current fold
#         ax.set_title(f"Fold {n+1}: R2={float(metric_list[n]):.4f}")

#     plt.suptitle(f"{method_name}: reals vs. predicteds")

#     reals_vs_preds_fig.tight_layout()

#     # plot all real vs. predicted values
#     together_reals_v_preds_fig, ax1 = plt.subplots()
#     ax1.scatter(all_reals, all_preds, marker="o")

#     # plot x=y line as a dashed line
#     ax1.plot(
#         [0, 1],
#         [0, 1],
#         color="r",
#         linestyle="--",
#         alpha=0.75,
#         zorder=0,
#         transform=ax1.transAxes,
#     )
#     ax1.set_title(f"{method_name}: R2={float(overall_metric):.4f}")
#     together_reals_v_preds_fig.tight_layout()

#     return [reals_vs_preds_fig, together_reals_v_preds_fig]


# def k_fold_confusion_matrix(
#     reals_list,
#     preds_list,
#     overall_metric,  # overall metric for all folds
#     method_name,  # name of method
#     metric_list,  # list of metrics for each fold
#     params,
# ):
#     # confusion matrix big subplots for each fold with AUC and acc in the top
#     # one confusion matrix which has all the fold-vals in it
#     # return both
#     pass


# def compare_methods_boxplot(performances, params, plot_type, rep_n=None):
#     """
#     Creates a boxplot to compare performance metrics of different methods.

#     Parameters
#     ----------
#     performances: dict
#         Dictionary of performance metrics for different methods.
#     params: dict
#         Additional parameters.
#     plot_type: str
#         Type of plot ('kfold' or 'repetition').
#     rep_n: int, optional
#         Repetition number (for repetition plots), by default None.

#     Returns
#     -------
#     fig: matplotlib.figure.Figure
#         The figure containing the boxplot.
#     """

#     # get method names and metric names
#     method_names = list(performances.keys())  # [method1name, method2name,...]
#     metric_names = list(
#         list(performances.values())[0].keys()
#     )  # [metric1name, metric2name]

#     # get metric values for each method
#     metric_1_values = [performances[method][metric_names[0]] for method in method_names]
#     metric_2_values = [performances[method][metric_names[1]] for method in method_names]

#     # create figure 1x2 subplots
#     fig, ax = plt.subplots(1, 2)
#     ax[0].grid()
#     ax[1].grid()

#     # create violin plots for each metric
#     bp = ax[0].violinplot(metric_1_values, vert=False, showmeans=True)

#     def set_violin_colors(instance, colour):
#         for pc in instance["bodies"]:
#             pc.set_facecolor(colour)
#             pc.set_edgecolor("black")
#             pc.set_alpha(0.5)
#         instance["cmeans"].set_edgecolor("black")
#         instance["cmins"].set_edgecolor("black")
#         instance["cmaxes"].set_edgecolor("black")
#         instance["cbars"].set_edgecolor("black")

#     set_violin_colors(bp, "violet")

#     ax[0].yaxis.set_ticks(np.arange(len(method_names)) + 1)
#     ax[0].set_yticklabels(method_names)
#     ax[0].get_xaxis().tick_bottom()
#     ax[0].set_xlim(right=1.0)

#     bp2 = ax[1].violinplot(metric_2_values, vert=False, showmeans=True)
#     set_violin_colors(bp2, "powderblue")

#     ax[1].yaxis.set_ticks(np.arange(len(method_names)) + 1)
#     ax[1].set_yticklabels([] * len(metric_2_values))
#     ax[1].get_xaxis().tick_bottom()

#     # set titles and limits
#     ax[0].set_title(metric_names[0])
#     ax[1].set_title(metric_names[1])
#     ax[1].set_xlim(left=0.0)

#     # set figure title based on plot type
#     if plot_type == "kfold":
#         plt.suptitle(
#             f"Distribution of metrics between cross-validation folds in repetition {rep_n}"
#         )
#     elif plot_type == "repetition":
#         plt.suptitle("Distribution of metrics between repetitions")
#     plt.tight_layout()

#     return fig


# def eval_replications(repetition_performances, params):
#     """
#     Evaluates and visualizes the results of multiple replications.

#     Parameters
#     ----------
#     repetition_performances: dict
#         Dictionary of performance metrics for different methods and replications.
#     params: dict
#         Additional parameters.
#     """

#     if params["num_replications"] > 1:
#         # Create a violin plot for repetition performances
#         # repetition performances structure: {method: {metric: [] for metric in metric_names} for
#         # method in methods}
#         repetition_violin_plot = compare_methods_boxplot(
#             repetition_performances, params, plot_type="repetition"
#         )

#         if params["cluster"] is False and params["log"] is False:
#             # save the plot as a png locally
#             plt.savefig(f"{params['local_fig_path']}/repetition_violin", dpi=180)
#             plt.close(repetition_violin_plot)
#         elif params["log"]:
#             # log the plot to wandb
#             wandb.log({"repetition_violin": wandb.Image(repetition_violin_plot)})
#             plt.close(repetition_violin_plot)
#             wandb.finish()
#     else:
#         print("Only one repetition, no repetition violin plot")
#         if params["log"]:
#             wandb.finish()


# def eval_one_rep_kfold(k_fold_performances, rep_number, params):
#     """
#     Evaluates and visualizes the results of one repetition in k-fold cross-validation.

#     Parameters
#     ----------
#     k_fold_performances: dict
#         Dictionary of performance metrics for different methods and folds.
#     rep_number: int
#         Repetition number.
#     params: dict
#         Additional parameters.
#     """
#     if params["kfold_flag"]:
#         # Create a violin plot for k-fold performances
#         kfold_violin_plot = compare_methods_boxplot(
#             k_fold_performances, params, plot_type="kfold", rep_n=rep_number
#         )

#         if params["cluster"] is False and params["log"] is False:
#             # save the plot as a png locally
#             plt.savefig(
#                 f"{params['local_fig_path']}/kfold_violin_rep{rep_number}", dpi=180
#             )
#             plt.close(kfold_violin_plot)
#         elif params["log"]:
#             # log the plot to wandb
#             wandb.log({f"kfold_violin_rep{rep_number}": wandb.Image(kfold_violin_plot)})
#             plt.close(kfold_violin_plot)
#             if rep_number != params["num_replications"] - 1:
#                 wandb.finish()


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
