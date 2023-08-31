"""
Edge correlation GNN model.
"""

import torch.nn as nn
from fusionlibrary.fusion_models.base_pl_model import ParentFusionModel
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class EdgeCorrGNN(ParentFusionModel, nn.Module):
    """
    Graph neural network with the edge weighting as the first tabular modality correlations and
    the node features as the second tabular modality features.

    Attributes
    ----------
    method_name : str
        Name of the method.
    modality_type : str
        Type of modality.
    fusion_type : str
        Type of fusion.
    graph_maker : function
        Function that creates the graph data structure.
    conv1 : GCNConv
        Graph convolutional layer. The first layer takes in the number of features of the
        second tabular modality as input.
    conv2 : GCNConv
        Graph convolutional layer. The second layer takes in 64 features as input.
    conv3 : GCNConv
        Graph convolutional layer. The third layer takes in 128 features as input.
    conv4 : GCNConv
        Graph convolutional layer. The fourth layer takes in 256 features as input.
    final_prediction : nn.Sequential
        Sequential layer containing the final prediction layers. The final prediction layers
        take in 256 features.

    Methods
    -------
    forward(x)
        Forward pass of the model.

    """

    method_name = "Edge Correlation GNN"
    modality_type = "both_tab"
    fusion_type = "graph"

    def __init__(self, pred_type, data_dims, params):
        """
        Parameters
        ----------
        pred_type : str
            Type of prediction to be performed.
        data_dims : dict
            Dictionary containing the dimensions of the data.
        params : dict
            Dictionary containing the parameters of the model.
        """

        ParentFusionModel.__init__(self, pred_type, data_dims, params)

        self.pred_type = pred_type

        self.graph_maker = EdgeCorrGraphMaker

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
        """
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
        x : torch.Tensor
            Tensor containing the tabular data.

        Returns
        -------
        list
            List containing the output of the model.
        """
        x_n, edge_index, edge_attr = x

        for layer in self.graph_conv_layers:
            x_n = layer(x_n, edge_index, edge_attr)
            x_n = x_n.relu()
            x_n = F.dropout(x_n, p=self.dropout_prob, training=self.training)

        out = self.final_prediction(x_n)

        return [
            out,
        ]


class EdgeCorrGraphMaker:
    def __init__(self, dataset):
        self.dataset = dataset

        self.threshold = 0.8  # how correlated the nodes need to be to be connected

    def make_graph(self):
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


# def edgecorr_graph_maker(dataset):
#     """
#     Creates the graph data structure for the edge correlation GNN model.

#     Parameters
#     ----------
#     dataset : torch.utils.data.Dataset
#         Dataset containing the tabular data.

#     Returns
#     -------
#     data : torch_geometric.data.Data
#         Graph data structure containing the tabular data.
#     """
#     tab1 = dataset[:][0]
#     tab2 = dataset[:][1]
#     labels = dataset[:][2]

#     num_nodes = tab1.shape[0]

#     # correlation matrix between nodes' tab1 features
#     corr_matrix = torch.corrcoef(tab1) - torch.eye(num_nodes)

#     threshold = 0.8  # how correlated the nodes need to be to be connected

#     edge_indices = np.where(np.abs(corr_matrix) >= threshold)
#     edge_indices = np.stack(edge_indices, axis=0)

#     # print("Number of edges: ", edge_indices.shape[1])

#     x = tab2
#     edge_index = torch.tensor(edge_indices, dtype=torch.long)
#     edge_attr = (
#         corr_matrix[edge_indices[0], edge_indices[1]] + 1
#     )  # add 1 to make all edge_attr positive

#     data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=labels)

#     return data
