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


# creating a new Plotter class for plotting the results of one model or multiple models


class Plotter:
    def __init__(self, trained_model_dict, params):
        super().__init__()

        self.trained_model_dict = trained_model_dict
        self.params = params
        self.pred_type = self.params["pred_type"]  # for less verbose access

        self.model_names = list(trained_model_dict.keys())
        print("model names:", self.model_names)

        for value in trained_model_dict.values():
            if isinstance(value, list):
                print("kfold!")
                print("num_k:", len(value))
                break

        # if there is one model
        if len(self.model_names) == 1:
            print("only one model, no comparison needed")
        else:
            print(
                "multiple models, comparison needed:", len(self.model_names), "models"
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
            "regression": None,  # csv saving? bar chart?
            "binary": None,
            "multiclass": None,
        }

        self.multi_model_kfold_plots = {
            "regression": None,  # violin plots of the folds
            "binary": None,
            "multiclass": None,
        }

    def get_kfold_numbers(self):
        self.train_reals = []
        self.train_preds = []
        self.val_reals = []
        self.val_preds = []
        self.kfold_plot_val_accs = []

        self.trained_model_list = self.trained_model_dict[self.current_model_name]

        for k in range(self.params["num_k"]):
            k_trained_model_var = vars(self.trained_model_list[k])
            self.train_reals.append(k_trained_model_var["train_reals"])
            self.train_preds.append(k_trained_model_var["train_preds"])
            self.val_reals.append(k_trained_model_var["val_reals"])
            self.val_preds.append(k_trained_model_var["val_preds"])

            self.kfold_plot_val_accs.append(
                self.trained_model_list[k]
                .metrics[self.pred_type][0]["metric"]
                .to(self.trained_model_list[k].device)(
                    k_trained_model_var["val_preds"],
                    k_trained_model_var["val_preds"],
                )
            )

        # altogether vals
        self.all_preds = torch.cat(self.val_preds)
        self.all_reals = torch.cat(self.val_reals)

        self.plot_val_acc = (
            self.trained_model_list[0]
            .metrics[self.pred_type][0]["metric"]
            .to(self.trained_model_list[0].device)(self.all_preds, self.all_reals)
        )

    def get_train_test_numbers(self):
        self.trained_model = self.trained_model_dict[self.current_model_name]

        # get lists of train_reals, train_preds, val_reals, val_preds and save to self
        self.train_reals = vars(self.trained_model_dict[self.model_names[0]])[
            "train_reals"
        ]
        self.train_preds = vars(self.trained_model_dict[self.model_names[0]])[
            "train_preds"
        ]
        self.val_reals = vars(self.trained_model_dict[self.model_names[0]])["val_reals"]
        self.val_preds = vars(self.trained_model_dict[self.model_names[0]])["val_preds"]

        self.train_end_metrics = [
            vars(self.trained_model_dict[self.model_names[0]])["metric1"],
            vars(self.trained_model_dict[self.model_names[0]])["metric2"],
        ]

        print("metrics:", self.train_end_metrics)

        self.plot_val_acc = self.trained_model.metrics[self.pred_type][0]["metric"].to(
            self.trained_model.device
        )(self.val_preds, self.val_reals)

    def plot_all(self):
        """
        Plots all the results.
        """
        if len(self.model_names) == 1:
            self.current_model_name = self.model_names[0]

            if self.params["kfold_flag"]:
                # get lists of train_reals, train_preds, val_reals, val_preds for all folds and save to self

                self.get_kfold_numbers()

                results_figs = self.single_model_kfold_plots[self.pred_type]()

                return results_figs

            else:
                self.get_train_test_numbers()

                self.trained_model = self.trained_model_dict[self.model_names[0]]

                # plot the results for one train/test model
                results_fig = self.single_model_tt_plots[self.pred_type]()

                return results_fig

        else:
            if self.params["kfold_flag"]:
                comparing_models_metrics = (
                    {}
                )  # keys: model names, values: lists of kfold metrics len(k)
                # or maybe value is another dictionary {"R2": [...,...,...], "MAE": [...,..,...]}

                for model_name in self.model_names:
                    self.current_model_name = model_name

                    self.get_kfold_numbers()
                    # get lists of train_reals, train_preds, val_reals, val_preds for all folds and save to self
                    # get altogether list of val_reals and val_preds over folds and save to self
                    # get metrics to put in the plot titles, add to metrics list

                    # plot the predefined kfold models for self.params["pred_type"]
                    #   (they will also take in the altogether list of val_reals and val_preds
                    #   as well as the lists of train_reals, train_preds, val_reals, val_preds)

                    # save etc

                # plot comparison of all kfold models: violin plot?

            else:
                comparing_models_metrics = (
                    {}
                )  # keys: model names, values: metric values (single numbers)
                # or maybe value is another dictionary {"R2": ... "MAE": ...}
                for model_name in self.model_names:
                    # get train_reals, train_preds, val_reals, val_preds and save to self
                    self.current_model_name = model_name
                    self.get_train_test_numbers()

                    # plot the predefined models for self.params["pred_type"]

                    # save figures to wandb or locally, or plt.show if we can for an example notebook?
                    pass

                # plot comparison of all train/test models: bar chart?

    def reals_vs_preds(self):
        """
        Plots the real values against the predicted values for the training and validation sets.
        """

        # JUST FOR ONE MODEL!

        fig, ax = plt.subplots()

        ax.scatter(self.train_reals, self.train_preds, c="blue", label="Train")
        ax.scatter(self.val_reals, self.val_preds, c="red", label="Validation")
        # val_acc = self.trained_model.metrics[self.params["pred_type"]][0]["metric"].to(
        #     self.trained_model.device
        # )(self.val_preds, self.val_reals)

        # Get the limits of the current scatter plot
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()

        # Set up data points for the x=y line
        line_x = np.linspace(min(x_min, y_min), max(x_max, y_max), 100)
        line_y = line_x

        # Plot the x=y line as a dashed line
        plt.plot(line_x, line_y, linestyle="dashed", color="black", label="x=y Line")

        ax.set_title(
            f"Validation {self.trained_model.metrics[self.params['pred_type']][0]['name']}: \
                {float(self.plot_val_acc):.4f}"
        )

        ax.set_xlabel("Real Values")
        ax.set_ylabel("Predictions")
        ax.legend()

        return fig

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
            f"Validation {self.trained_model.metrics[self.pred_type][0]['name']}: \
                {float(self.plot_val_acc):.4f}"
        )

        plt.tight_layout()

        return fig

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
            ax.set_title(f"Fold {n+1}: R2={float(self.kfold_plot_val_accs[n]):.4f}")

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
        ax1.set_title(f"{self.current_model_name}: R2={float(self.plot_val_acc):.4f}")
        together_reals_v_preds_fig.tight_layout()

        # TODO combine these images to be pretty without worrying about what k is? ask chatgpt
        return [reals_vs_preds_fig, together_reals_v_preds_fig]

    def confusion_matrix_plotter_kfold(self):
        """
        Plots the confusion matrix of a kfold model.
        """
        pass

    def compare_violin_plot(self):
        """
        Plots a violin plot comparing the results of multiple models.
        """
        pass


def reals_vs_preds(model, train_reals, train_preds, val_reals, val_preds):
    """
    Plots the real values against the predicted values for the training and validation sets.

    Parameters
    ----------

    model: torch.nn.Module
        The model to be evaluated.
    train_reals: torch.Tensor
        The real values for the training set.
    train_preds: torch.Tensor
        The predicted values for the training set.
    val_reals: torch.Tensor
        The real values for the validation set.
    val_preds: torch.Tensor
        The predicted values for the validation set.

    Returns
    -------

    fig: matplotlib.figure.Figure
        The figure containing the plot.
    """
    fig, ax = plt.subplots()

    ax.scatter(train_reals, train_preds, c="blue", label="Train")
    ax.scatter(val_reals, val_preds, c="red", label="Validation")
    val_acc = model.metrics[model.pred_type][0]["metric"].to(model.device)(
        val_preds, val_reals
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
        f"Validation {model.metrics[model.pred_type][0]['name']}: {np.round(val_acc,3)}"
    )

    ax.set_xlabel("Real Values")
    ax.set_ylabel("Predictions")
    ax.legend()

    return fig


def confusion_matrix_plot(model, train_reals, train_preds, val_reals, val_preds):
    """
    Plots the confusion matrix.

    Parameters
    ----------
    model: torch.nn.Module
        The model to be evaluated.
    train_reals: torch.Tensor
        The real values for the training set.
    train_preds: torch.Tensor
        The predicted values for the training set.
    val_reals: torch.Tensor
        The real values for the validation set.
    val_preds: torch.Tensor
        The predicted values for the validation set.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The figure containing the confusion matrix plot.
    """

    # Get the confusion matrix for the validation set
    conf_matrix = confusion_matrix(y_true=val_reals, y_pred=val_preds)

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    # Plot the confusion matrix as a heatmap
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            # Add the value of each cell to the plot
            ax.text(
                x=j, y=i, s=conf_matrix[i, j], va="center", ha="center", size="xx-large"
            )

    plt.xlabel("Predictions", fontsize=18)
    plt.ylabel("Actuals", fontsize=18)

    # calculate the accuracy of the validation set using the model's metric
    val_acc = model.metrics[model.pred_type][0]["metric"].to(model.device)(
        val_preds, val_reals
    )

    plt.title(
        f"Validation {model.metrics[model.pred_type][0]['name']}: {np.round(val_acc,3)}"
    )

    plt.tight_layout()

    return fig


def k_fold_reals_vs_preds(
    reals_list,
    preds_list,
    overall_metric,  # overall metric for all folds
    method_name,  # name of method
    metric_list,  # list of metrics for each fold
    params,
):
    """
    Plots real values against predicted values for each fold in k-fold cross-validation.

    Parameters
    ----------
    reals_list: list of torch.Tensor
        List of real values for each fold.
    preds_list: list of torch.Tensor
        List of predicted values for each fold.
    overall_metric: float
        Overall metric for all folds.
    method_name: str
        Name of the method.
    metric_list: list of float
        List of metrics for each fold.
    params: dict
        Additional parameters.

    Returns
    -------
    fig_list: list of matplotlib.figure.Figure
        List of figures containing the plots.
    """

    # concatenate real and pred values from all folds
    all_preds = torch.cat(reals_list)
    all_reals = torch.cat(preds_list)

    N = params["num_k"]
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
        reals = reals_list[n]
        preds = preds_list[n]

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
        ax.set_title(f"Fold {n+1}: R2={float(metric_list[n]):.4f}")

    plt.suptitle(f"{method_name}: reals vs. predicteds")

    reals_vs_preds_fig.tight_layout()

    # plot all real vs. predicted values
    together_reals_v_preds_fig, ax1 = plt.subplots()
    ax1.scatter(all_reals, all_preds, marker="o")

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
    ax1.set_title(f"{method_name}: R2={float(overall_metric):.4f}")
    together_reals_v_preds_fig.tight_layout()

    return [reals_vs_preds_fig, together_reals_v_preds_fig]


def k_fold_confusion_matrix(
    reals_list,
    preds_list,
    overall_metric,  # overall metric for all folds
    method_name,  # name of method
    metric_list,  # list of metrics for each fold
    params,
):
    # confusion matrix big subplots for each fold with AUC and acc in the top
    # one confusion matrix which has all the fold-vals in it
    # return both
    pass


def compare_methods_boxplot(performances, params, plot_type, rep_n=None):
    """
    Creates a boxplot to compare performance metrics of different methods.

    Parameters
    ----------
    performances: dict
        Dictionary of performance metrics for different methods.
    params: dict
        Additional parameters.
    plot_type: str
        Type of plot ('kfold' or 'repetition').
    rep_n: int, optional
        Repetition number (for repetition plots), by default None.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The figure containing the boxplot.
    """

    # get method names and metric names
    method_names = list(performances.keys())  # [method1name, method2name,...]
    metric_names = list(
        list(performances.values())[0].keys()
    )  # [metric1name, metric2name]

    # get metric values for each method
    metric_1_values = [performances[method][metric_names[0]] for method in method_names]
    metric_2_values = [performances[method][metric_names[1]] for method in method_names]

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
    ax[0].set_title(metric_names[0])
    ax[1].set_title(metric_names[1])
    ax[1].set_xlim(left=0.0)

    # set figure title based on plot type
    if plot_type == "kfold":
        plt.suptitle(
            f"Distribution of metrics between cross-validation folds in repetition {rep_n}"
        )
    elif plot_type == "repetition":
        plt.suptitle("Distribution of metrics between repetitions")
    plt.tight_layout()

    return fig


def eval_replications(repetition_performances, params):
    """
    Evaluates and visualizes the results of multiple replications.

    Parameters
    ----------
    repetition_performances: dict
        Dictionary of performance metrics for different methods and replications.
    params: dict
        Additional parameters.
    """

    if params["num_replications"] > 1:
        # Create a violin plot for repetition performances
        # repetition performances structure: {method: {metric: [] for metric in metric_names} for
        # method in methods}
        repetition_violin_plot = compare_methods_boxplot(
            repetition_performances, params, plot_type="repetition"
        )

        if params["cluster"] is False and params["log"] is False:
            # save the plot as a png locally
            plt.savefig(f"{params['local_fig_path']}/repetition_violin", dpi=180)
            plt.close(repetition_violin_plot)
        elif params["log"]:
            # log the plot to wandb
            wandb.log({"repetition_violin": wandb.Image(repetition_violin_plot)})
            plt.close(repetition_violin_plot)
            wandb.finish()
    else:
        print("Only one repetition, no repetition violin plot")
        if params["log"]:
            wandb.finish()


def eval_one_rep_kfold(k_fold_performances, rep_number, params):
    """
    Evaluates and visualizes the results of one repetition in k-fold cross-validation.

    Parameters
    ----------
    k_fold_performances: dict
        Dictionary of performance metrics for different methods and folds.
    rep_number: int
        Repetition number.
    params: dict
        Additional parameters.
    """
    if params["kfold_flag"]:
        # Create a violin plot for k-fold performances
        kfold_violin_plot = compare_methods_boxplot(
            k_fold_performances, params, plot_type="kfold", rep_n=rep_number
        )

        if params["cluster"] is False and params["log"] is False:
            # save the plot as a png locally
            plt.savefig(
                f"{params['local_fig_path']}/kfold_violin_rep{rep_number}", dpi=180
            )
            plt.close(kfold_violin_plot)
        elif params["log"]:
            # log the plot to wandb
            wandb.log({f"kfold_violin_rep{rep_number}": wandb.Image(kfold_violin_plot)})
            plt.close(kfold_violin_plot)
            if rep_number != params["num_replications"] - 1:
                wandb.finish()


def plot_graph(graph_data, params):
    """
    Plots the graph structure.

    Parameters
    ----------
    graph_data: pyg.data.Data
        Graph data.
    params: dict
        Additional parameters.
    """

    # convert graph data to networkx graph
    G = pyg.utils.convert.to_networkx(
        graph_data,
        to_undirected=True,
        remove_self_loops=True,
        node_attrs=["x"],
        edge_attrs=["edge_attr"],
    )
    for u, v, d in G.edges(data=True):
        d["weight"] = d["edge_attr"]
        # d['nodecolor'] = d["y"].item()

    # colour nodes with their label
    for n, d in G.nodes(data=True):
        d["nodecolor"] = graph_data.y[n].item()

    edges, weights = zip(*nx.get_edge_attributes(G, "weight").items())
    nodes, colors = zip(*nx.get_node_attributes(G, "nodecolor").items())

    pos = nx.spectral_layout(G)

    # draw graph with node colour and edge weight
    nx.draw(
        G,
        pos,
        node_color=colors,
        edgelist=edges,
        edge_color=weights,
        width=5.0,
        edge_cmap=plt.cm.Blues,
        cmap=plt.cm.coolwarm,
        node_size=50,
    )
    vmin = np.min(weights)
    vmax = np.max(weights)
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.Blues, norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm)

    if params["cluster"] is False and params["log"] is False:
        # save graph structure plot as a png locally
        plt.savefig(f"{params['local_fig_path']}/graph_structure", dpi=180)
        plt.close()


def visualise_graphspace(h, color, params, path_suffix, method_name):
    """
    Visualizes the graph space using t-SNE.

    Parameters
    ----------
    h: torch.Tensor
        Graph data.
    color: list
        List of colors.
    params: dict
        Additional parameters.
    path_suffix: str
        Suffix for the saved image.
    method_name: str
        Name of the method.

    Returns
    -------
    fig: matplotlib.figure.Figure
        The figure containing the t-SNE plot.
    """

    z = TSNE(n_components=2, perplexity=30).fit_transform(
        h.reshape(-1, 1).detach().cpu().numpy()
    )

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_xticks([])
    ax.set_yticks([])

    scatter = ax.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    cbar = fig.colorbar(scatter)
    cbar.set_label("Color Legend")
    ax.set_title(f"Method: {method_name}")
    return fig
