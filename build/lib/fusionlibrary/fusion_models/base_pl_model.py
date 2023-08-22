"""
Base lightning module for all fusion models and parent class for all fusion models.
"""

import os

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
import torchmetrics as tm
import wandb
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

# from fusionlibrary.eval_functions import (
#     confusion_matrix_plot,
#     k_fold_confusion_matrix,
#     k_fold_reals_vs_preds,
#     reals_vs_preds,
#     visualise_graphspace,
# )


class BaseModel(pl.LightningModule):
    """
    Base model for all fusion models

    Attributes
    ----------
    model : class
        Fusion model class.
    pred_type : str
        Type of prediction to be made. Options: binary, multiclass, regression.
    multiclass_dim : int
        Number of classes for multiclass prediction.
    fusion_type : str
        Type of fusion to be used. Options: early, late, graph.
    kfold_flag : bool
        Whether to use k-fold cross validation.
    modality_type : str
        Type of modality fusion to be used. Options: operation, subspace, graph, tensor, attention, Uni-modal
    method_name : str
        Name of the method.
    subspace_method : str
        Subspace method to be used - specific to the subspace fusion method
    graph_maker : str
        Graph maker to be used - specific to the graph fusion method
    params : dict
        Dictionary of parameters.
    train_mask : tensor
        Mask for training data - used for the graph fusion methods
    val_mask : tensor
        Mask for validation data - used for the graph fusion methods
    loss_functions : dict
        Dictionary of loss functions.
    output_activation_functions : dict
        Dictionary of output activation functions.
    metrics : dict
        Dictionary of metrics.
    eval_plots : dict
        Dictionary of evaluation plots.
    kfold_plots : dict
        Dictionary of k-fold plots.

    Methods
    -------
    safe_squeeze(tensor)
        Squeeze tensor if it is not 1D.
    get_data_from_batch(batch, train=True)
        Get data from batch.
    get_model_outputs(x)
        Get model outputs.
    get_model_outputs_and_loss(x, y, train=True)
        Get model outputs and loss.
    training_step(batch, batch_idx)
        Training step.
    validation_step(batch, batch_idx)
        Validation step.
    configure_optimizers()
        Configure optimizers.
    plot_eval_figs(train_loader, val_loader, path_suffix)
        Plot evaluation figures.
    plot_kfold_eval_figs(all_kf_val_preds, all_kf_val_reals, kfold_metrics, path_suffix)
        Plot k-fold evaluation figures.
    """

    def __init__(self, model):
        """
        Parameters
        ----------
        model : class
            Fusion model class.

        Returns
        -------
        None
        """
        super().__init__()
        self.model = model
        self.pred_type = model.pred_type
        self.multiclass_dim = model.multiclass_dim
        self.fusion_type = model.fusion_type
        self.kfold_flag = model.kfold_flag
        self.modality_type = model.modality_type
        self.method_name = model.method_name
        if hasattr(model, "subspace_method"):
            self.subspace_method = model.subspace_method
        else:
            self.subspace_method = None

        if hasattr(model, "graph_maker"):
            self.graph_maker = model.graph_maker
        # else:
        #     self.graph_maker = None # <-- don't know if this is necessary yet

        self.params = model.params

        if hasattr(model, "train_mask") is False:
            self.train_mask = None
            self.val_mask = None

        self.loss_functions = {
            "binary": lambda logits, y: F.binary_cross_entropy_with_logits(
                logits, y.float()
            ),
            "multiclass": lambda logits, y: F.cross_entropy(
                self.safe_squeeze(logits), self.safe_squeeze(y)
            ),
            "regression": F.mse_loss,
        }

        self.output_activation_functions = {
            "binary": torch.round,
            "multiclass": lambda x: torch.argmax(nn.Softmax(dim=-1)(x), dim=-1),
            "regression": lambda x: x,
        }

        # metrics
        self.metrics = {
            "binary": [
                {
                    "metric": tm.AUROC(task="binary"),
                    "name": "binary_auroc",
                },  # needs logits
                {"metric": tm.Accuracy(task="binary"), "name": "binary_accuracy"},
            ],
            "multiclass": [
                {
                    "metric": tm.AUROC(
                        task="multiclass", num_classes=self.multiclass_dim
                    ),
                    "name": "multiclass_auroc",  # needs logits
                },
                {
                    "metric": tm.Accuracy(
                        task="multiclass", num_classes=self.multiclass_dim, top_k=1
                    ),
                    "name": "multiclass_accuracy"
                    # Add additional metrics for multiclass classification if needed
                },
            ],
            "regression": [
                {
                    "metric": tm.R2Score(),
                    "name": "R2"
                    # Add additional metrics for regression if needed
                },
                {"metric": tm.MeanAbsoluteError(), "name": "MAE"},
            ],
        }
        if self.pred_type not in self.metrics:
            raise ValueError(f"Unsupported pred_type: {self.pred_type}")

        # self.eval_plots = {
        #     "regression": [reals_vs_preds],
        #     "binary": [confusion_matrix_plot],
        #     "multiclass": [confusion_matrix_plot],
        # }

        # self.kfold_plots = {
        #     "regression": k_fold_reals_vs_preds,
        #     "binary": k_fold_confusion_matrix,
        #     "multiclass": k_fold_confusion_matrix,
        # }

        self.metric_names_list = [
            metric["name"] for metric in self.metrics[self.pred_type]
        ]

        # storing the final validation reals and preds
        self.batch_val_reals = []
        self.batch_val_preds = []
        self.batch_val_logits = []
        self.batch_train_reals = []
        self.batch_train_preds = []
        self.batch_train_logits = []

    def safe_squeeze(self, tensor):
        """
        Squeeze tensor if it is not 1D.

        Parameters
        ----------
        tensor : tensor
            Tensor to be squeezed.

        Returns
        -------
        tensor
            Squeezed tensor.
        """
        # Check if the tensor is 1D, in which case, no squeezing is needed
        if len(tensor.shape) == 1:
            return tensor
        # Otherwise, remove the first dimension
        else:
            return tensor.squeeze(dim=0)

    def get_data_from_batch(self, batch, train=True):
        """
        Get data from batch.

        Parameters
        ----------
        batch : tensor
            Batch of data.
        train : bool
            Whether the data is training data.

        Returns
        -------
        x : tensor
            Input data.
        y : tensor
            Labels.
        """
        if self.fusion_type == "graph":
            x = (batch.x, batch.edge_index, batch.edge_attr)
            y = batch.y
        else:
            if len(batch) == 2:
                x, y = batch
            elif len(batch) == 3:
                x1, x2, y = batch
                x = (x1, x2)
            else:
                raise ValueError(
                    "Batch size is not 2 (preds and labels) or 3 (2 pred data types and labels)\
                    modalities long"
                )

        return x, y

    def get_model_outputs(self, x):
        """
        Get model outputs.

        Parameters
        ----------
        x : tensor
            Input data.

        Returns
        -------
        logits : tensor
            Logits.
        reconstructions : tensor
            Reconstructions (returned if the model has a custom loss function such as a subspace
                method)

        note:
        if you get an error here, check that the forward output is [out,] or [out, reconstructions]
        """
        model_outputs = self.model(x)

        logits, *reconstructions = model_outputs
        logits = logits.squeeze(dim=1)

        return logits, reconstructions

    def get_model_outputs_and_loss(self, x, y, train=True):
        """
        Get model outputs and loss.

        Parameters
        ----------
        x : tensor
            Input data.
        y : tensor
            Labels.

        Returns
        -------
        loss : tensor
            Loss.
        end_output : tensor
            Final output.
        logits : tensor
            Logits.
        """
        logits, reconstructions = self.get_model_outputs(x)

        end_output = self.output_activation_functions[self.pred_type](logits)

        # if we're doing graph-based fusion and train/test doesn't work the same as normal
        if hasattr(self, "train_mask"):
            if train:
                logits = logits[self.train_mask]
                y = y[self.train_mask]
                end_output = end_output[self.train_mask]
            else:
                logits = logits[self.val_mask]
                y = y[self.val_mask]
                end_output = end_output[self.val_mask]

        loss = self.loss_functions[self.pred_type](logits, y)

        if reconstructions != [] and self.model.custom_loss is not None:
            added_loss = self.model.custom_loss(
                reconstructions[0], x[-1]
            )  # x[-1] bc img is always last
            loss += added_loss

        return loss, end_output, logits

    # def on_train_start(self):
    #     # Clear the stored validation values at the beginning of training
    #     self.val_reals = []
    #     self.val_preds = []

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Parameters
        ----------
        batch : tensor
            Batch of data.
        batch_idx : int
            Batch index.

        Returns
        -------
        loss : tensor
            Loss.
        """
        # if batch_idx == 0:
        # Clear the stored training values at the beginning of training again
        # we don't want to save the old values from the previous epoch,
        # only final validation epoch
        # self.batch_train_reals = []
        # self.batch_train_preds = []
        # self.batch_train_logits = []
        # self.batch_val_preds = []
        # self.batch_val_reals = []
        # self.batch_val_logits = []

        x, y = self.get_data_from_batch(batch)

        loss, end_output, logits = self.get_model_outputs_and_loss(x, y)

        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        for metric in self.metrics[self.pred_type]:
            if "auroc" in metric["name"]:
                predicted = logits
            else:
                predicted = end_output

            if self.safe_squeeze(predicted).shape[0] == 1:
                # if it's a single value, we can't calculate a metric
                pass
            else:
                train_step_acc = metric["metric"].to(self.device)(
                    self.safe_squeeze(predicted),
                    self.safe_squeeze(y[self.train_mask])
                    # .squeeze(),  # we can do self.train_mask even if it's None bc y is a tensor
                )

                self.log(
                    metric["name"] + "_train",
                    train_step_acc,
                    logger=True,
                    on_epoch=True,
                    on_step=False,
                )

        # Store real and predicted values for training

        self.batch_train_reals.append(self.safe_squeeze(y[self.train_mask]).detach())
        self.batch_train_preds.append(predicted.detach())
        self.batch_train_logits.append(logits.detach())

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Parameters
        ----------
        batch : tensor
            Batch of data.
        batch_idx : int
            Batch index.

        Returns
        -------
        None
        """
        x, y = self.get_data_from_batch(batch, train=False)

        loss, end_output, logits = self.get_model_outputs_and_loss(x, y, train=False)

        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        # for i, metric in enumerate(self.metrics[self.pred_type]):
        #     if "auroc" in metric["name"]:
        #         predicted = logits
        #     else:
        #         predicted = end_output

        #     val_step_acc = metric["metric"].to(self.device)(
        #         self.safe_squeeze(predicted), self.safe_squeeze(y[self.val_mask])
        #     )

        #     # print("what's used to calc the end metric:")
        #     # print("preds:", self.safe_squeeze(predicted))
        #     # print("reals:", self.safe_squeeze(y[self.val_mask]))
        #     # print("reals without mask:", self.safe_squeeze(y))

        #     self.log(
        #         metric["name"] + "_val",
        #         val_step_acc,
        #         logger=True,
        #         on_epoch=True,
        #     )

        # Store real and predicted values for later access

        self.batch_val_reals.append(self.safe_squeeze(y[self.val_mask]).detach())
        self.batch_val_preds.append(end_output.detach())
        self.batch_val_logits.append(self.safe_squeeze(logits).detach())
        # print(self.safe_squeeze(logits).detach().shape)

    def validation_epoch_end(self, outputs):
        """
        Gets the final validation epoch outputs and metrics.
        When metrics are calculated at the validation step and logged on on_epoch=True,
        the batch metrics are averaged. However, some metrics don't average well (e.g. R2).
        Therefore, we're calculating the final validation metrics here on the full validation set.
        """

        self.val_reals = torch.cat(self.batch_val_reals, dim=-1)
        self.val_preds = torch.cat(self.batch_val_preds, dim=-1)
        # print(self.batch_val_logits[0])
        # print(self.batch_val_logits[1])
        # print(len(self.batch_val_logits))
        self.val_logits = torch.cat(self.batch_val_logits, dim=0)

        try:
            self.train_reals = torch.cat(self.batch_train_reals, dim=-1)
            self.train_preds = torch.cat(self.batch_train_preds, dim=-1)
        except:
            pass

        for i, metric in enumerate(self.metrics[self.pred_type]):
            if "auroc" in metric["name"]:
                predicted = self.val_logits
            else:
                predicted = self.val_preds

            val_step_acc = metric["metric"].to(self.device)(
                self.safe_squeeze(predicted),
                self.safe_squeeze(self.val_reals),
            )

            self.log(
                metric["name"] + "_val",
                val_step_acc,
                logger=True,
                on_epoch=True,
            )

        self.batch_val_reals = []
        self.batch_val_preds = []
        self.batch_val_logits = []
        self.batch_train_reals = []
        self.batch_train_preds = []
        self.batch_train_logits = []

    def configure_optimizers(self):
        """
        Configure optimizers.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    # def get_reals_and_preds(self, train_loader, val_loader):
    #     """
    #     Gets real and predicted values for train and validation data.

    #     Parameters
    #     ----------
    #     train_loader : DataLoader
    #         Training data loader.
    #     val_loader : DataLoader
    #         Validation data loader.

    #     """
    #     self.model.eval()

    #     train_reals = []
    #     train_preds = []
    #     val_reals = []
    #     val_preds = []

    #     for batch in train_loader:
    #         x, y = self.get_data_from_batch(batch, train=True)

    #         logits, reconstructions = self.get_model_outputs(x)
    #         outputs = self.output_activation_functions[self.pred_type](logits)

    #         if self.train_mask is not None:
    #             # if it's a graph, we can visualise the final graph space
    #             outputs = outputs[self.train_mask]
    #             y = y[self.train_mask]

    #         train_reals.append(y.detach())
    #         train_preds.append(outputs.detach())

    #     for batch in val_loader:
    #         x, y = self.get_data_from_batch(batch, train=False)
    #         logits, reconstructions = self.get_model_outputs(x)
    #         outputs = self.output_activation_functions[self.pred_type](logits)

    #         if self.val_mask is not None:
    #             outputs = outputs[self.val_mask]
    #             y = y[self.val_mask]

    #         val_reals.append(y.detach())
    #         val_preds.append(outputs.detach())

    #     self.train_reals = torch.cat(train_reals)
    #     self.train_preds = torch.cat(train_preds)
    #     self.val_reals = torch.cat(val_reals)
    #     self.val_preds = torch.cat(val_preds)

    # def plot_eval_figs(self, train_loader, val_loader, path_suffix):
    #     """
    #     Plot evaluation figures.

    #     Parameters
    #     ----------
    #     train_loader : DataLoader
    #         Training data loader.
    #     val_loader : DataLoader
    #         Validation data loader.
    #     path_suffix : str
    #         Path suffix.

    #     Returns
    #     -------
    #     fig_outputs : list
    #         List of figures.
    #     val_reals : list
    #         List of validation real values.
    #     val_preds : list
    #         List of validation predicted values.
    #     """
    #     self.model.eval()

    #     train_reals = []
    #     train_preds = []
    #     val_reals = []
    #     val_preds = []
    #     fig_outputs = []

    #     for batch in train_loader:
    #         x, y = self.get_data_from_batch(batch, train=True)

    #         logits, reconstructions = self.get_model_outputs(x)
    #         # logits = self.model(x).squeeze(dim=1)
    #         outputs = self.output_activation_functions[self.pred_type](logits)

    #         if self.train_mask is not None:
    #             # if it's a graph, we can visualise the final graph space
    #             if self.params["pred_type"] != "regression":
    #                 graphspace_fig = visualise_graphspace(
    #                     outputs, y, self.params, path_suffix, self.method_name
    #                 )

    #             outputs = outputs[self.train_mask]
    #             y = y[self.train_mask]

    #         train_reals.append(y.detach())
    #         train_preds.append(outputs.detach())

    #     for batch in val_loader:
    #         x, y = self.get_data_from_batch(batch, train=False)
    #         logits, reconstructions = self.get_model_outputs(x)
    #         outputs = self.output_activation_functions[self.pred_type](logits)

    #         if self.val_mask is not None:
    #             outputs = outputs[self.val_mask]
    #             y = y[self.val_mask]

    #         val_reals.append(y.detach())
    #         val_preds.append(outputs.detach())

    #     train_reals = torch.cat(train_reals)
    #     train_preds = torch.cat(train_preds)
    #     val_reals = torch.cat(val_reals)
    #     val_preds = torch.cat(val_preds)

    #     for fig_type in self.eval_plots[self.pred_type]:
    #         fig = fig_type(self, train_reals, train_preds, val_reals, val_preds)
    #         fig_outputs.append(fig)

    #     if self.params["pred_type"] != "regression":
    #         graphspace_fig = visualise_graphspace(
    #             torch.cat([train_preds, val_preds]),
    #             torch.cat([train_reals, val_reals]),
    #             self.params,
    #             path_suffix,
    #             self.method_name,
    #         )

    #     if self.params["log"] is False:
    #         for fig_type, fig in zip(self.eval_plots[self.pred_type], fig_outputs):
    #             fig.savefig(
    #                 f"{self.params['local_fig_path']}/{fig_type.__name__}{path_suffix}",
    #                 dpi=180,
    #             )
    #             plt.close()

    #         if self.params["pred_type"] != "regression":
    #             graphspace_fig.savefig(
    #                 f"{self.params['local_fig_path']}/graphspace{path_suffix}",
    #                 dpi=180,
    #             )
    #             plt.close()

    #     if isinstance(self.logger, pl_loggers.TensorBoardLogger):
    #         # if it's not wandb logger
    #         # self.logger.experiment.add_image(fig_type.__name__ + path_suffix, fig, 0)
    #         pass
    #     elif isinstance(self.logger, pl_loggers.WandbLogger):
    #         # self.logger.experiment.log(
    #         #     {fig_type.__name__ + path_suffix: wandb.Image(fig)}
    #         # )
    #         pass

    #     return (fig_outputs, val_reals, val_preds)

    # def plot_kfold_eval_figs(
    #     self, all_kf_val_preds, all_kf_val_reals, kfold_metrics, path_suffix
    # ):
    #     """
    #     Plot k-fold evaluation figures.

    #     Parameters
    #     ----------
    #     all_kf_val_preds : list
    #         List of all k-fold validation predicted values.
    #     all_kf_val_reals : list
    #         List of all k-fold validation real values.
    #     kfold_metrics : list
    #         List of k-fold metrics.
    #     path_suffix : str
    #         Path suffix.

    #     Returns
    #     -------
    #     list
    #         List of overall metrics.
    #     """
    #     # TODO add options for multiclass and binary
    #     self.model.eval()

    #     all_preds = torch.cat(all_kf_val_preds)
    #     all_reals = torch.cat(all_kf_val_reals)

    #     overall_metric = self.metrics[self.pred_type][0]["metric"](all_preds, all_reals)
    #     overall_metric_2 = self.metrics[self.pred_type][1]["metric"](
    #         all_preds, all_reals
    #     )

    #     figures = self.kfold_plots[self.pred_type](
    #         reals_list=all_kf_val_reals,
    #         preds_list=all_kf_val_preds,
    #         overall_metric=overall_metric,
    #         method_name=self.model.method_name,
    #         metric_list=kfold_metrics,
    #         params=self.model.params,
    #     )

    #     # figures = k_fold_reals_vs_preds(
    #     #     reals_list=all_kf_val_reals,
    #     #     preds_list=all_kf_val_preds,
    #     #     overall_metric=overall_metric,
    #     #     method_name=self.model.method_name,
    #     #     metric_list=kfold_metrics,
    #     #     params=self.model.params,
    #     # )

    #     figure_names = ["real_vs_pred_per_fold", "overall_kfold_real_vs_pred"]
    #     if figures is not None:
    #         for i, figure in enumerate(figures):
    #             if isinstance(self.logger, pl_loggers.TensorBoardLogger):
    #                 # if it's not wandb logger
    #                 # self.logger.experiment.add_image(fig_type.__name__ + path_suffix, fig, 0)
    #                 pass
    #             elif isinstance(self.logger, pl_loggers.WandbLogger):
    #                 self.logger.experiment.log(
    #                     {figure_names[i] + path_suffix: wandb.Image(figure)}
    #                 )
    #                 plt.close(figure)

    #     return [overall_metric, overall_metric_2]


class ParentFusionModel:
    """
    Parent class for all fusion models.

    Attributes
    ----------
    pred_type : str
        Type of prediction to be made. Options: binary, multiclass, regression.
    data_dims : list
        List of data dimensions.
    mod1_dim : int
        Dimension of modality 1.
    mod2_dim : int
        Dimension of modality 2.
    img_dim : tuple
        Dimensions of image modality.
    params : dict
        Dictionary of parameters.
    multiclass_dim : int
        Number of classes for multiclass prediction.
    kfold_flag : bool
        Whether to use k-fold cross validation.

    Methods
    -------
    set_final_pred_layers(input_dim=64)
        Set final prediction layers.
    set_mod1_layers()
        Set layers for modality 1.
    set_mod2_layers()
        Set layers for modality 2.
    set_img_layers()
        Set layers for image modality.
    set_fused_layers(fused_dim)
        Set layers for fused modality.
    """

    def __init__(self, pred_type, data_dims, params):
        """
        Parameters
        ----------
        pred_type : str
            Type of prediction to be made. Options: binary, multiclass, regression.
        data_dims : list
            List of data dimensions.
        params : dict
            Dictionary of parameters.
        """
        super().__init__()
        self.pred_type = pred_type
        self.data_dims = data_dims
        self.mod1_dim = self.data_dims[0]
        self.mod2_dim = self.data_dims[1]
        self.img_dim = self.data_dims[2]
        self.params = params
        self.multiclass_dim = params["multiclass_dims"]
        self.kfold_flag = params["kfold_flag"]
        # self.subspace_method = None

    def set_final_pred_layers(self, input_dim=64):
        """
        Set final prediction layers.

        Parameters
        ----------
        input_dim : int
            Input dimension to final layers - may depend on fusion configuration.

        Returns
        -------
        None
        """
        # final predictions
        if self.pred_type == "binary":
            self.final_prediction = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid())

        elif self.pred_type == "multiclass":
            self.final_prediction = nn.Sequential(
                nn.Linear(input_dim, self.multiclass_dim)
                # , nn.Softmax(dim=-1)
            )

        elif self.pred_type == "regression":
            self.final_prediction = nn.Sequential(nn.Linear(input_dim, 1))

    def set_mod1_layers(self):
        """
        Set layers for modality 1

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.mod1_layers = nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(self.mod1_dim, 32),
                    nn.ReLU(),
                ),
                "layer 2": nn.Sequential(
                    nn.Linear(32, 64),
                    nn.ReLU(),
                ),
                "layer 3": nn.Sequential(
                    nn.Linear(64, 128),
                    nn.ReLU(),
                ),
                "layer 4": nn.Sequential(
                    nn.Linear(128, 256),
                    nn.ReLU(),
                ),
                "layer 5": nn.Sequential(
                    nn.Linear(256, 256),
                    nn.ReLU(),
                ),
            }
        )

    def set_mod2_layers(self):
        """
        Set layers for modality 2

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.mod2_layers = nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(self.mod2_dim, 32),
                    nn.ReLU(),
                ),
                "layer 2": nn.Sequential(
                    nn.Linear(32, 64),
                    nn.ReLU(),
                ),
                "layer 3": nn.Sequential(
                    nn.Linear(64, 128),
                    nn.ReLU(),
                ),
                "layer 4": nn.Sequential(
                    nn.Linear(128, 256),
                    nn.ReLU(),
                ),
                "layer 5": nn.Sequential(
                    nn.Linear(256, 256),
                    nn.ReLU(),
                ),
            }
        )

    def set_img_layers(self):
        """
        Set layers for image modality

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.img_layers = nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool3d((2, 2, 2)),
                ),
                "layer 2": nn.Sequential(
                    nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool3d((2, 2, 2)),
                ),
                "layer 3": nn.Sequential(
                    nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool3d((2, 2, 2)),
                ),
                "layer 4": nn.Sequential(
                    nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool3d((2, 2, 2)),
                ),
                "layer 5": nn.Sequential(
                    nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool3d((2, 2, 2)),
                ),
            }
        )

    def set_fused_layers(self, fused_dim):
        """
        Set layers for fused modality

        Parameters
        ----------
        fused_dim : int
            Dimension of fused modality: how many features are there after fusion?
                e.g. if we have 2 modalities with 64 features each, and the fusion method
                    was concatenation, the fused_dim would be 128

        Returns
        -------
        None
        """
        self.fused_layers = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.15),
        )
