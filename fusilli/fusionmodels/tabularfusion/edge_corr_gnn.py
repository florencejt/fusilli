"""
Edge correlation GNN model: edges are weighted by the correlation between the nodes' first tabular modality features.
"""

import torch.nn as nn
from fusilli.fusionmodels.base_model import ParentFusionModel
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from fusilli.utils import check_model_validity


class EdgeCorrGraphMaker:
    """
    Creates the graph data structure for the edge correlation GNN model.

    Attributes
    ----------
    dataset : torch.utils.data.Dataset
        Dataset containing the tabular data.
    threshold : float
        How correlated the nodes need to be to be connected. Default: 0.8
    """

    def __init__(self, dataset):
        """
        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset containing the tabular data.
        """
        self.dataset = dataset

        self.threshold = 0.8  # how correlated the nodes need to be to be connected

    def check_params(self):
        """
        Checks the parameters of the model.

        Returns
        -------
        None
        """

        if self.threshold < 0 or self.threshold > 1:
            raise ValueError(
                (
                    "Incorrect attribute range: The threshold must be between 0 and 1, "
                    f"inclusive. The threshold is currently: {self.threshold}"
                )
            )

        check_model_validity.check_dtype(self.threshold, float, "threshold")

    def make_graph(self):
        """
        Creates the graph data structure.

        Returns
        -------
        data : torch_geometric.data.Data
            Graph data structure containing the tabular data.
        """

        self.check_params()
        tab1 = self.dataset[:][0]
        tab2 = self.dataset[:][1]
        labels = self.dataset[:][2]

        num_nodes = tab1.shape[0]

        # correlation matrix between nodes' tab1 features
        corr_matrix = torch.corrcoef(tab1) - torch.eye(num_nodes)

        edge_indices = np.where(np.abs(corr_matrix) >= self.threshold)
        edge_indices = np.stack(edge_indices, axis=0)

        # print("Number of edges: ", edge_indices.shape[1])

        x = tab2
        edge_index = torch.tensor(edge_indices, dtype=torch.long)
        edge_attr = (
                corr_matrix[edge_indices[0], edge_indices[1]] + 1
        )  # add 1 to make all edge_attr positive

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=labels)

        return data


class EdgeCorrGNN(ParentFusionModel, nn.Module):
    """
    Graph neural network with the edge weighting as the first tabular modality correlations and
    the node features as the second tabular modality features.

    Attributes
    ----------
    graph_maker : function
        Function that creates the graph data structure: :class:`~.EdgeCorrGraphMaker`
    graph_conv_layers : nn.Sequential
        Sequential layer containing the graph convolutional layers.
    dropout_prob : float
        Dropout probability. Default: 0.5
    final_prediction : nn.Sequential
        Sequential layer containing the final prediction layers. The final prediction layers
        take in 256 features.
    """

    #: str: Name of the method.
    method_name = "Edge Correlation GNN"
    #: str: Type of modality.
    modality_type = "tabular_tabular"
    #: str: Type of fusion.
    fusion_type = "graph"
    # class: Graph maker class.
    graph_maker = EdgeCorrGraphMaker

    def __init__(self, prediction_task, data_dims, multiclass_dimensions):
        """
        Parameters
        ----------
        prediction_task : str
            Type of prediction to be performed.
        data_dims : list
            List containing the dimensions of the data.
        multiclass_dimensions : int
            Number of classes in the multiclass classification task.
        """
        ParentFusionModel.__init__(self, prediction_task, data_dims, multiclass_dimensions)

        self.prediction_task = prediction_task

        self.graph_conv_layers = nn.Sequential(
            GCNConv(self.mod2_dim, 64),
            GCNConv(64, 128),
            GCNConv(128, 256),
            GCNConv(256, 256),
        )

        self.dropout_prob = 0.5

        self.calc_fused_layers()

    def calc_fused_layers(self):
        """
        Calculates the number of features after the fusion layer.

        Returns
        -------
        None
        """

        if self.dropout_prob < 0 or self.dropout_prob > 1:
            raise ValueError(
                (
                    "Incorrect attribute range: The dropout probability must be between,"
                    f" 0 and 1, inclusive. The dropout probability is currently: {self.dropout_prob}"
                )
            )

        check_model_validity.check_dtype(
            self.graph_conv_layers, nn.Sequential, "graph_conv_layers"
        )
        check_model_validity.check_dtype(self.dropout_prob, float, "dropout_prob")

        # make sure the first layer takes in the number of features of the second tabular modality
        self.graph_conv_layers[0] = GCNConv(
            self.mod2_dim, self.graph_conv_layers[0].out_channels
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

        # ~~ Checks ~~
        # check x is a tuple of length 3
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
