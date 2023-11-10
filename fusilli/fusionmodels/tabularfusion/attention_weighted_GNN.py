"""
Inspired by the multi-modal brain age paper:
- Pretraining a model on concatenated tabular modalities
- Getting out the attention weights from the model: what does this mean?
- Make the graph structure:
- node features are second modality,
- edges are based on the attention weights: get weighted phenotypes by multiplying the attention weights by the multiple-
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
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch import Trainer


class AttentionWeightMLP(pl.LightningModule):

    def __init__(self, pred_type, data_dims, multiclass_dim):
        super().__init__()

        self.pred_type = pred_type
        self.multiclass_dim = multiclass_dim

        self.mod1_dim = data_dims[0]
        self.mod2_dim = data_dims[1]

        self.fc1 = nn.Linear(self.mod1_dim + self.mod2_dim, 200)
        self.fc2 = nn.Linear(200, 400)
        self.fc3 = nn.Linear(400, 400)
        self.fc4 = nn.Linear(400, 200)
        self.fc5 = nn.Linear(200, self.mod1_dim + self.mod2_dim)

        ParentFusionModel.set_fused_layers(self, self.mod1_dim + self.mod2_dim)

        ParentFusionModel.set_final_pred_layers(self)

    def forward(self, x):
        x = torch.cat(x, dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)

        attention_weights = torch.sigmoid(x)

        out_fused_layers = self.fused_layers(x)

        out_pred = self.final_prediction(out_fused_layers)

        return out_pred, attention_weights

    def training_step(self, batch, batch_idx):
        x1, x2, y = batch
        y_hat, weights = self.forward((x1, x2))
        loss = F.mse_loss(y_hat.squeeze(), y.squeeze())
        self.log('train_loss', loss, logger=None)
        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, y = batch
        y_hat, weights = self.forward((x1, x2))
        loss = F.mse_loss(y_hat.squeeze(), y.squeeze())
        self.log('val_loss', loss, logger=None)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def create_attention_weights(self, x):
        preds, weights = self.forward(x)
        return weights


class AttentionWeightedGraphMaker:

    def __init__(self, dataset):
        self.dataset = dataset

    def check_params(self):
        pass

    def make_graph(self):
        # get out the tabular data
        all_labels = self.dataset[:][2]

        # split the dataset
        [train_dataset, test_dataset] = torch.utils.data.random_split(
            self.dataset, [1 - 0.2, 0.2]
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

        # infer the pred type from the dataset?
        if torch.is_floating_point(all_labels[0]):
            pred_type = "regression"
            multiclass_dim = None
        else:
            if len(np.unique(all_labels)) == 2:
                pred_type = "binary"
                multiclass_dim = None
            else:
                pred_type = "multiclass"
                multiclass_dim = len(np.unique(all_labels))

        # initialise the MLP model
        self.MLP_instance = AttentionWeightMLP(pred_type, data_dims, multiclass_dim)

        num_nodes = all_labels.shape[0]  # number of nodes/subjects

        # set up a pytorch trainer
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        val_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=15,
            verbose=False,
            mode="min",
        )

        callbacks_list = [early_stop_callback]

        self.trainer = Trainer(
            num_sanity_val_steps=0,
            callbacks=callbacks_list,
            log_every_n_steps=2,
            logger=None,
            enable_checkpointing=True,
        )
        # fit the MLP model
        self.trainer.fit(self.MLP_instance, train_dataloader, val_dataloader)
        self.trainer.validate(self.MLP_instance, val_dataloader)

        # get out the train attention weights
        train_attention_weights = self.MLP_instance.create_attention_weights(
            (train_dataset[:][0], train_dataset[:][1])
        )
        # get out the validation attention weights
        val_attention_weights = self.MLP_instance.create_attention_weights(
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
        print("all_weighted_phenotypes", all_weighted_phenotypes.shape)

        # get probability of each edge from weighted phenotypes
        distances = torch.cdist(all_weighted_phenotypes, all_weighted_phenotypes) ** 2

        # normalise to go between 0 and 1
        distances = distances / torch.max(distances)
        print(distances)
        distances = distances.detach().numpy()
        probs = np.exp(-distances)
        # take away the identity
        probs = probs - np.eye(probs.shape[0])
        print(probs)
        print("max prob: ", np.max(probs))
        print("min prob: ", np.min(probs))
        print("mean prob: ", np.mean(probs))

        top_quartile = np.percentile(probs, 75)
        print("top quartile: ", top_quartile)

        edge_indices = np.where(probs > top_quartile)
        edge_indices = np.stack(edge_indices, axis=0)

        print("Number of edges: ", edge_indices.shape[1])

        # make the node features the second modality (train and val)
        node_features = torch.cat((tab2_train, tab2_test), dim=0)
        # construct the graph structure
        edge_index = torch.tensor(edge_indices, dtype=torch.long)

        edge_attr = torch.tensor(distances[edge_indices[0], edge_indices[1]])

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=all_labels)

        return data


class AttentionWeightedGNN(ParentFusionModel, nn.Module):
    method_name = "Attention-weighted GNN"
    modality_type = "tabular_tabular"
    fusion_type = "graph"
    graph_maker = AttentionWeightedGraphMaker

    def __init__(self, pred_type, data_dims, params):
        ParentFusionModel.__init__(self, pred_type, data_dims, params)

        self.pred_type = pred_type

        # self.graph_conv_layers = nn.Sequential(
        #     GCNConv(self.mod2_dim, 64),
        #     GCNConv(64, 128),
        #     GCNConv(128, 256),
        #     GCNConv(256, 256),
        # )

        self.graph_conv_layers = nn.Sequential(
            ChebConv(self.mod2_dim, 64, K=3),
            ChebConv(64, 128, K=3),
            ChebConv(128, 256, K=3),
            ChebConv(256, 256, K=3),
        )

        self.fused_dim = self.graph_conv_layers[-1].out_channels
        self.set_final_pred_layers(self.fused_dim)

        self.dropout_prob = 0.2

    def forward(self, x):
        x_n, edge_index, edge_attr = x

        for layer in self.graph_conv_layers:
            x_n = layer(x_n, edge_index, edge_attr)
            x_n = x_n.relu()
            # x_n = F.dropout(x_n, p=self.dropout_prob, training=self.training)

        out = self.final_prediction(x_n)

        return [
            out,
        ]
