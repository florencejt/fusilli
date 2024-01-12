"""
Attention weighted GNN model: the edge weights are the attention weights from a pre-trained MLP and the node features are the second modality.
"""
import torch.nn as nn
from fusilli.fusionmodels.base_model import ParentFusionModel
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, ChebConv
import torch.nn.functional as F
from fusilli.utils import check_model_validity
import lightning.pytorch as pl
from fusilli.utils.training_utils import (
    get_checkpoint_filenames_for_subspace_models,
    init_trainer,
)
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch import Trainer


class AttentionWeightMLP(pl.LightningModule):
    """
    MLP based on ConcatTabularData for the attention weighted GNN.

    Attributes
    ----------
    prediction_task : str
        Type of prediction to be performed.
    multiclass_dimensions : int
        Number of classes for multiclass classification. If not multiclass classification, this is None.
    mod1_dim : int
        Number of features of the first modality.
    mod2_dim : int
        Number of features of the second modality.
    weighting_layers : nn.ModuleDict
        Module dictionary containing the weighting layers. The layers must have input size of the
        first modality dimension plus the second modality dimension and output size of the first modality dimension
        plus the second modality dimension.
    fused_dim : int
        Number of features of the fused layers. This is the final output shape of the weighting layers.
    fused_layers : nn.Sequential
        Sequential layer containing the fused layers. Calculated in the :meth:`~ParentFusionModel.set_fused_layers` method.
    final_prediction : nn.Sequential
        Sequential layer containing the final prediction layers. The final prediction layers take in the number of
        features of the fused layers as input. Calculated in the :meth:`~ParentFusionModel.set_final_pred_layers` method.
    """

    def __init__(self, prediction_task, data_dims, multiclass_dimensions):
        """

        Parameters
        ----------
        prediction_task : str
            Type of prediction to be performed.
        data_dims : list
            List containing the dimensions of the data.
        multiclass_dimensions : int
            Number of classes for multiclass classification. If not multiclass classification, this is None.
        """
        super().__init__()

        self.prediction_task = prediction_task
        self.multiclass_dimensions = multiclass_dimensions

        self.mod1_dim = data_dims[0]
        self.mod2_dim = data_dims[1]

        self.weighting_layers = nn.ModuleDict(
            {
                "Layer 1": nn.Sequential(nn.Linear(self.mod1_dim + self.mod2_dim, 256),
                                         nn.ReLU()),
                "Layer 2": nn.Sequential(nn.Linear(256, 128),
                                         nn.ReLU()),
                "Layer 3": nn.Sequential(nn.Linear(128, 128),
                                         nn.ReLU()),
                "Layer 4": nn.Sequential(nn.Linear(128, 256),
                                         nn.ReLU()),
                "Layer 5": nn.Sequential(nn.Linear(256, self.mod1_dim + self.mod2_dim),
                                         nn.ReLU()),
            }
        )

        self.fused_dim = self.mod1_dim + self.mod2_dim
        ParentFusionModel.set_fused_layers(self, self.fused_dim)

        self.calc_fused_layers()

    def calc_fused_layers(self):
        """
        Checks the parameters of the model, sets the final prediction layers and calculates the fused layers based on
        any modifications to the model.
        """
        check_model_validity.check_dtype(self.weighting_layers, nn.ModuleDict, "weighting_layers")

        self.fused_dim = self.mod1_dim + self.mod2_dim

        # check final layer output size is the same as the fused dim
        final_weighting_layer = self.weighting_layers[list(self.weighting_layers.keys())[-1]][0]

        if final_weighting_layer.out_features != self.fused_dim:
            raise ValueError(
                (
                    "Incorrect attribute range: The final weighting_layer layer must have an output size of"
                    f" {self.fused_dim} (the same as the input). The final weighting layer output size is currently: {final_weighting_layer.out_features}"
                )
            )

        self.fused_layers, out_dim = check_model_validity.check_fused_layers(
            self.fused_layers, self.fused_dim
        )

        ParentFusionModel.set_final_pred_layers(self, out_dim)

    def forward(self, x):
        """

        Parameters
        ----------
        x: tuple
            Tuple containing the two modalities input data.

        Returns
        -------
        out_pred: torch.Tensor
            Prediction output of the model.
        attention_weights: torch.Tensor
            Attention weights of the model. Final layer of the model sigmoided.

        """

        x = torch.cat(x, dim=1).to(torch.float32)

        for layer in self.weighting_layers.values():
            x = layer(x)

        attention_weights = torch.sigmoid(x)

        out_fused_layers = self.fused_layers(x)

        out_pred = self.final_prediction(out_fused_layers)

        return out_pred, attention_weights

    def training_step(self, batch, batch_idx):
        """
        Training step of the model.

        Parameters
        ----------
        batch: tuple
            Tuple containing the two modalities input data and the labels.
        batch_idx: int
            Index of the batch.

        Returns
        -------
        loss: torch.Tensor
            Loss of the model.

        """
        x1, x2, y = batch

        y_hat, weights = self.forward((x1, x2))

        if self.prediction_task == "multiclass":
            # turn the labels into one hot vectors
            y = F.one_hot(y, num_classes=self.multiclass_dimensions).to(torch.float32)

        loss = F.mse_loss(y_hat.squeeze(), y.to(torch.float32).squeeze())
        self.log('train_loss', loss, logger=None)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step of the model.

        Parameters
        ----------
        batch: tuple
            Tuple containing the two modalities input data and the labels.
        batch_idx: int
            Index of the batch.

        Returns
        -------
        loss: torch.Tensor
            Loss of the model.

        """
        x1, x2, y = batch
        y_hat, weights = self.forward((x1, x2))

        if self.prediction_task == "multiclass":
            # turn the labels into one hot vectors
            y = F.one_hot(y, num_classes=self.multiclass_dimensions).to(torch.float32)

        loss = F.mse_loss(y_hat.squeeze(), y.to(torch.float32).squeeze())
        self.log('val_loss', loss, logger=None)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimiser of the model.

        Returns
        -------
        optimiser: torch.optim
            Optimiser of the model.

        """
        optimiser = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimiser

    def create_attention_weights(self, x):
        """
        Create the attention weights of the model for a given input.

        Parameters
        ----------
        x: tuple
            Tuple containing the two modalities input data.

        Returns
        -------
        weights: torch.Tensor
            Attention weights of the model. Final layer of the model sigmoided.

        """
        preds, weights = self.forward(x)
        return weights


class AttentionWeightedGraphMaker:
    """
    Class to make the graph structure for the attention weighted GNN.

    Attributes
    ----------
    dataset: Dataset
        Dataset containing the tabular data.
    early_stop_callback: EarlyStopping
        Early stopping callback for the MLP model.
    edge_probability_threshold: int
        Probability threshold for the edges of the graph. e.g. 75 means the edges associated with the top 25% of probabilities are used.
    attention_MLP_test_size: float
        Test size for the MLP model.
    max_epochs: int
        Maximum number of epochs for the MLP model. Default -1.
    AttentionWeightingMLPInstance: AttentionWeightMLP
        Instance of the MLP model.
    trainer: Trainer
        Trainer of the model.
    train_idxs: list
        List of the indices of the training data.
    test_idxs: list
        List of the indices of the test data.

    """

    def __init__(self, dataset):
        """

        Parameters
        ----------
        dataset: Dataset
            Dataset containing the tabular data.
        """
        self.dataset = dataset

        self.early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=15,
            verbose=False,
            mode="min",
        )

        self.edge_probability_threshold = 75

        self.attention_MLP_test_size = 0.2

        # initialise MLP
        data_dims = [self.dataset[:][0].shape[1], self.dataset[:][1].shape[1]]

        if torch.is_floating_point(self.dataset[:][2][0]):
            prediction_task = "regression"
            multiclass_dim = None
        else:
            if len(np.unique(self.dataset[:][2])) == 2:
                prediction_task = "binary"
                multiclass_dim = None
            else:
                prediction_task = "multiclass"
                multiclass_dim = len(np.unique(self.dataset[:][2]))

        self.AttentionWeightingMLPInstance = AttentionWeightMLP(prediction_task, data_dims, multiclass_dim)

        self.max_epochs = -1

    def check_params(self):
        """
        Checks the parameters of the model.

        """

        # check the distance threshold percentage is an int between 0 and 100
        check_model_validity.check_dtype(self.edge_probability_threshold, int, "edge_probability_threshold")
        if self.edge_probability_threshold <= 0 or self.edge_probability_threshold > 100:
            raise ValueError(
                (
                    "Incorrect attribute range: The distance_threshold_percentage must be between 0 and 100, "
                    f"inclusive. The threshold is currently: {self.edge_probability_threshold}"
                )
            )

        # check early stopping is an EarlyStopping object
        check_model_validity.check_dtype(self.early_stop_callback, EarlyStopping, "early_stop_callback")

        # check attention MLP test size is a float between 0 and 1
        if self.attention_MLP_test_size <= 0 or self.attention_MLP_test_size > 1:
            raise ValueError(
                (
                    "Incorrect attribute range: The attention_MLP_test_size must be between 0 and 1, "
                    f"inclusive. The threshold is currently: {self.attention_MLP_test_size}"
                )
            )
        check_model_validity.check_dtype(self.attention_MLP_test_size, float, "attention_MLP_test_size")

    def make_graph(self):
        """
        Make the graph structure for the attention weighted GNN.

        Returns
        -------
        data: Data
            Data object containing the graph structure.

        """
        # get out the tabular data
        all_labels = self.dataset[:][2]

        tab1 = self.dataset[:][0]
        tab2 = self.dataset[:][1]
        labels = self.dataset[:][2]

        # split the dataset
        [train_dataset, test_dataset] = torch.utils.data.random_split(
            self.dataset, [1 - self.attention_MLP_test_size, self.attention_MLP_test_size]
        )

        self.train_idxs = train_dataset.indices
        self.test_idxs = test_dataset.indices

        # get the dataset
        tab1_train = train_dataset[:][0]
        tab2_train = train_dataset[:][1]
        labels_train = train_dataset[:][2]

        tab1_test = test_dataset[:][0]
        tab2_test = test_dataset[:][1]
        labels_test = test_dataset[:][2]

        data_dims = [tab1_train.shape[1], tab2_train.shape[1]]
        num_nodes = all_labels.shape[0]  # number of nodes/subjects

        # set up a pytorch trainer
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        val_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        callbacks_list = [self.early_stop_callback]

        self.trainer = Trainer(
            num_sanity_val_steps=0,
            callbacks=callbacks_list,
            log_every_n_steps=2,
            logger=False,
            enable_checkpointing=False,
            max_epochs=self.max_epochs,
        )

        # fit the MLP model
        self.trainer.fit(self.AttentionWeightingMLPInstance, train_dataloader, val_dataloader)
        self.trainer.validate(self.AttentionWeightingMLPInstance, val_dataloader)

        # get out the train attention weights
        train_attention_weights = self.AttentionWeightingMLPInstance.create_attention_weights(
            (train_dataset[:][0], train_dataset[:][1])
        )
        # get out the validation attention weights
        val_attention_weights = self.AttentionWeightingMLPInstance.create_attention_weights(
            (test_dataset[:][0], test_dataset[:][1])
        )

        # normalise the attention weights
        train_attention_weights = train_attention_weights / torch.sum(train_attention_weights)

        val_attention_weights = val_attention_weights / torch.sum(val_attention_weights)

        # make the weighted phenotypes: multiple data by attention weights
        # concatenate tab1 and tab2
        all_tab_train = torch.cat((tab1_train, tab2_train), dim=1)
        all_tab_val = torch.cat((tab1_test, tab2_test), dim=1)
        train_weighted_phenotypes = all_tab_train * train_attention_weights
        val_weighted_phenotypes = all_tab_val * val_attention_weights

        # concatenate the weighted phenotypes
        all_weighted_phenotypes = torch.cat((train_weighted_phenotypes, val_weighted_phenotypes), dim=0)

        # get probability of each edge from weighted phenotypes
        distances = torch.cdist(all_weighted_phenotypes, all_weighted_phenotypes) ** 2

        # normalise to go between 0 and 1
        distances = distances / torch.max(distances)
        distances = distances.detach().numpy()
        probs = np.exp(-distances)
        # take away the identity
        probs = probs - np.eye(probs.shape[0])

        top_percentage = np.percentile(probs, self.edge_probability_threshold)

        edge_indices = np.where(probs > top_percentage)
        edge_indices = np.stack(edge_indices, axis=0)

        # make the node features the second modality (train and val)
        node_features = torch.cat((tab2_train, tab2_test), dim=0)
        # construct the graph structure
        edge_index = torch.tensor(edge_indices, dtype=torch.long)

        edge_attr = torch.tensor(distances[edge_indices[0], edge_indices[1]])

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=all_labels)

        return data


class AttentionWeightedGNN(ParentFusionModel, nn.Module):
    """
    Graph neural network with the edge weighting as the distances between each nodes' weighted phenotypes and the node features as the second tabular modality features.

    This is a model inspired by method in `Bintsi et al. (2023) <https://arxiv.org/abs/2307.04639>`_ : *Multimodal brain age estimation using interpretable adaptive population-graph learning*.


    Attributes
    ----------
    prediction_task : str
        Type of prediction to be performed.
    graph_conv_layers : nn.Sequential
        Sequential layer containing the graph convolutional layers. By default ChebConv layers.
    fused_dim : int
        Number of features of the fused layers. This is the final output shape of the graph convolutional layers.
    final_prediction : nn.Sequential
        Sequential layer containing the final prediction layers. The final prediction layers

    """

    #: str: Name of the method.
    method_name = "Attention-weighted GNN"

    #: str: Type of modality.
    modality_type = "tabular_tabular"

    #: str: Type of fusion.
    fusion_type = "graph"

    # class: Graph maker class.
    graph_maker = AttentionWeightedGraphMaker

    def __init__(self, prediction_task, data_dims, multiclass_dimensions):
        """
        Parameters
        ----------
        prediction_task : str
            Type of prediction to be performed.
        data_dims : list
            List containing the dimensions of the data.
        multiclass_dimensions : int
            Number of classes for multiclass classification. If not multiclass classification, this is None.
        """

        ParentFusionModel.__init__(self, prediction_task, data_dims, multiclass_dimensions)

        self.prediction_task = prediction_task

        self.graph_conv_layers = nn.Sequential(
            ChebConv(self.mod2_dim, 64, K=3),
            ChebConv(64, 128, K=3),
            ChebConv(128, 256, K=3),
            ChebConv(256, 256, K=3),
        )

        self.dropout_prob = 0.2

        self.calc_fused_layers()

    def calc_fused_layers(self):
        """
        Checks the parameters of the model, sets the final prediction layers and calculates the fused layers based on any modifications to the model.
        """

        # check graph layers are sequential
        check_model_validity.check_dtype(self.graph_conv_layers, nn.Sequential, "graph_conv_layers")
        check_model_validity.check_dtype(self.dropout_prob, float, "dropout_prob")

        # check dropout probability is between 0 and 1
        if self.dropout_prob < 0 or self.dropout_prob > 1:
            raise ValueError(
                (
                    f"Incorrect attribute range: The dropout probability must be between, 0 and 1, inclusive. The dropout probability is currently: {self.dropout_prob}"
                )
            )

        self.fused_dim = self.graph_conv_layers[-1].out_channels
        self.set_final_pred_layers(self.fused_dim)

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : tuple
            Tuple containing the tabular data and the graph data structure:
            (node features, edge indices, edge attributes)

        Returns
        -------
        list
            List containing the output of the model.
        """

        check_model_validity.check_model_input(x, correct_length=3)

        x_n, edge_index, edge_attr = x

        for layer in self.graph_conv_layers:
            x_n = layer(x_n, edge_index, edge_attr)
            x_n = x_n.relu()
            x_n = F.dropout(x_n, p=self.dropout_prob, training=self.training)

        out = self.final_prediction(x_n)

        return [
            out,
        ]
