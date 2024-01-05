"""
These tests go through the whole pipeline and check that we can modify the model structure
and the subspace model structure and the from_new_data methods will still work without errors.
"""

import pytest
from fusilli.data import prepare_fusion_data
from fusilli.train import train_and_save_models
from fusilli.utils.model_chooser import import_chosen_fusion_models
from fusilli.eval import ConfusionMatrix
from ..test_data.test_TrainTestDataModule import create_test_files_more_features
import torch.nn as nn
from datetime import datetime
import os
import torch
from lightning.pytorch.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, ChebConv

layer_mods = {
    "ImgUnimodal": {
        "img_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Conv2d(1, 24, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
                "layer 2": nn.Sequential(
                    nn.Conv2d(24, 48, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
                "layer 3": nn.Sequential(
                    nn.Conv2d(48, 64, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
            }
        ),
        "fused_layers": nn.Sequential(
            nn.Linear(6400, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
    },
    "Tabular1Unimodal": {
        "mod1_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(10, 32),
                    nn.ReLU(),
                ),
                "layer 2": nn.Sequential(
                    nn.Linear(32, 66),
                    nn.ReLU(),
                ),
                "layer 3": nn.Sequential(
                    nn.Linear(66, 128),
                    nn.ReLU(),
                ),
            }
        ),
        "fused_layers": nn.Sequential(
            nn.Linear(80, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
    },
    "Tabular2Unimodal": {
        "mod2_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(15, 32),
                    nn.ReLU(),
                ),
                "layer 2": nn.Sequential(
                    nn.Linear(32, 66),
                    nn.ReLU(),
                ),
                "layer 3": nn.Sequential(
                    nn.Linear(66, 128),
                    nn.ReLU(),
                ),
            }
        ),
        "fused_layers": nn.Sequential(
            nn.Linear(25, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
    },
    "ConcatImgLatentTabDoubleLoss": {
        "latent_dim": 50,
        "fused_layers": nn.Sequential(
            nn.Linear(25, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
    },
    "ConcatImgLatentTabDoubleTrain": {
        "fused_layers": nn.Sequential(
            nn.Linear(25, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
    },
    "DAETabImgMaps": {
        "fusion_layers": nn.Sequential(
            nn.Linear(20, 420),
            nn.ReLU(),
            nn.Linear(420, 100),
            nn.ReLU(),
            nn.Linear(100, 78),
        ),
    },
    "denoising_autoencoder_subspace_method": {
        "autoencoder.latent_dim": 180,  # denoising autoencoder latent dim
        "autoencoder.upsampler": nn.Sequential(
            nn.Linear(25, 80),
            nn.ReLU(),
            nn.Linear(80, 100),
            nn.ReLU(),
            nn.Linear(100, 150),
            nn.ReLU(),
        ),
        "autoencoder.downsampler": nn.Sequential(
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, 80),
            nn.ReLU(),
            nn.Linear(80, 25),
            nn.ReLU(),
        ),
        "img_unimodal.img_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Conv2d(1, 40, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
                "layer 2": nn.Sequential(
                    nn.Conv2d(40, 60, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
                "layer 3": nn.Sequential(
                    nn.Conv2d(60, 85, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
            }
        ),
        "img_unimodal.fused_layers": nn.Sequential(
            nn.Linear(25, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
    },
    "concat_img_latent_tab_subspace_method": {
        "autoencoder.latent_dim": 180,  # img unimodal autoencoder
        "autoencoder.encoder": nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ),
        "autoencoder.decoder": nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
        ),
    },
    "ConcatImageMapsTabularData": {
        "img_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Conv2d(1, 35, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
                "layer 2": nn.Sequential(
                    nn.Conv2d(35, 70, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
                "layer 3": nn.Sequential(
                    nn.Conv2d(70, 128, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
            }
        ),
        "fused_layers": nn.Sequential(
            nn.Linear(25, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
    },
    "ConcatImageMapsTabularMaps": {
        "mod1_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(10, 32),
                    nn.ReLU(),
                ),
                "layer 2": nn.Sequential(
                    nn.Linear(32, 66),
                    nn.ReLU(),
                ),
                "layer 3": nn.Sequential(
                    nn.Linear(66, 128),
                    nn.ReLU(),
                ),
            }
        ),
        "img_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Conv2d(1, 35, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
                "layer 2": nn.Sequential(
                    nn.Conv2d(35, 70, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
                "layer 3": nn.Sequential(
                    nn.Conv2d(70, 128, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
            }
        ),
        "fused_layers": nn.Sequential(
            nn.Linear(25, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
    },
    "ConcatTabularData": {
        "fused_layers": nn.Sequential(
            nn.Linear(25, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
    },
    "ConcatTabularFeatureMaps": {
        "mod1_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(10, 32),
                    nn.ReLU(),
                ),
                "layer 2": nn.Sequential(
                    nn.Linear(32, 66),
                    nn.ReLU(),
                ),
                "layer 3": nn.Sequential(
                    nn.Linear(66, 128),
                    nn.ReLU(),
                ),
            }
        ),
        "mod2_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(15, 45),
                    nn.ReLU(),
                ),
                "layer 2": nn.Sequential(
                    nn.Linear(45, 70),
                    nn.ReLU(),
                ),
                "layer 3": nn.Sequential(
                    nn.Linear(70, 100),
                    nn.ReLU(),
                ),
            }
        ),
        "fused_layers": nn.Sequential(
            nn.Linear(25, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
    },
    "ImageDecision": {
        "mod1_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(10, 32),
                    nn.ReLU(),
                ),
                "layer 2": nn.Sequential(
                    nn.Linear(32, 66),
                    nn.ReLU(),
                ),
                "layer 3": nn.Sequential(
                    nn.Linear(66, 128),
                    nn.ReLU(),
                ),
            }
        ),
        "img_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Conv2d(1, 35, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
                "layer 2": nn.Sequential(
                    nn.Conv2d(35, 70, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
                "layer 3": nn.Sequential(
                    nn.Conv2d(70, 128, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
            }
        ),
    },
    "TabularDecision": {
        "mod1_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(10, 32),
                    nn.ReLU(),
                ),
                "layer 2": nn.Sequential(
                    nn.Linear(32, 66),
                    nn.ReLU(),
                ),
                "layer 3": nn.Sequential(
                    nn.Linear(66, 128),
                    nn.ReLU(),
                ),
            }
        ),
        "mod2_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(15, 45),
                    nn.ReLU(),
                ),
                "layer 2": nn.Sequential(
                    nn.Linear(45, 70),
                    nn.ReLU(),
                ),
                "layer 3": nn.Sequential(
                    nn.Linear(70, 100),
                    nn.ReLU(),
                ),
            }
        ),
    },
    "TabularChannelWiseMultiAttention": {
        "mod1_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(10, 32),
                    nn.ReLU(),
                ),
                "layer 2": nn.Sequential(
                    nn.Linear(32, 66),
                    nn.ReLU(),
                ),
                "layer 3": nn.Sequential(
                    nn.Linear(66, 128),
                    nn.ReLU(),
                ),
            }
        ),
        "mod2_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(15, 32),
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
            }
        ),
    },
    "ImageChannelWiseMultiAttention": {
        "mod1_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(10, 32),
                    nn.ReLU(),
                ),
                "layer 2": nn.Sequential(
                    nn.Linear(32, 66),
                    nn.ReLU(),
                ),
                "layer 3": nn.Sequential(
                    nn.Linear(66, 128),
                    nn.ReLU(),
                ),
            }
        ),
        "img_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Conv2d(1, 35, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
                "layer 2": nn.Sequential(
                    nn.Conv2d(35, 70, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
                "layer 3": nn.Sequential(
                    nn.Conv2d(70, 128, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
            }
        ),
    },
    "TabularCrossmodalAttention": {
        "mod1_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(10, 32),
                    nn.ReLU(),
                ),
                "layer 2": nn.Sequential(
                    nn.Linear(32, 66),
                    nn.ReLU(),
                ),
                "layer 3": nn.Sequential(
                    nn.Linear(66, 128),
                    nn.ReLU(),
                ),
            }
        ),
        "mod2_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(15, 32),
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
            }
        ),
    },
    "CrossmodalMultiheadAttention": {
        "mod1_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(10, 32),
                    nn.ReLU(),
                ),
                "layer 2": nn.Sequential(
                    nn.Linear(32, 66),
                    nn.ReLU(),
                ),
                "layer 3": nn.Sequential(
                    nn.Linear(66, 128),
                    nn.ReLU(),
                ),
            }
        ),
        "img_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Conv2d(1, 35, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
                "layer 2": nn.Sequential(
                    nn.Conv2d(35, 70, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
                "layer 3": nn.Sequential(
                    nn.Conv2d(70, 128, kernel_size=(3, 3), padding=0),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 2)),
                ),
            }
        ),
    },
    "ActivationFusion": {
        "mod1_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(10, 32),
                    nn.ReLU(),
                ),
                "layer 2": nn.Sequential(
                    nn.Linear(32, 66),
                    nn.ReLU(),
                ),
                "layer 3": nn.Sequential(
                    nn.Linear(66, 128),
                    nn.ReLU(),
                ),
            }
        ),
        "mod2_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(15, 45),
                    nn.ReLU(),
                ),
                "layer 2": nn.Sequential(
                    nn.Linear(45, 70),
                    nn.ReLU(),
                ),
                "layer 3": nn.Sequential(
                    nn.Linear(70, 128),
                    nn.ReLU(),
                ),
            }
        ),
        "fused_layers": nn.Sequential(
            nn.Linear(25, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
    },
    "AttentionAndSelfActivation": {
        "mod1_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(10, 32),
                    nn.ReLU(),
                ),
                "layer 2": nn.Sequential(
                    nn.Linear(32, 66),
                    nn.ReLU(),
                ),
                "layer 3": nn.Sequential(
                    nn.Linear(66, 128),
                    nn.ReLU(),
                ),
            }
        ),
        "mod2_layers": nn.ModuleDict(
            {
                "layer 1": nn.Sequential(
                    nn.Linear(15, 45),
                    nn.ReLU(),
                ),
                "layer 2": nn.Sequential(
                    nn.Linear(45, 70),
                    nn.ReLU(),
                ),
                "layer 3": nn.Sequential(
                    nn.Linear(70, 128),
                    nn.ReLU(),
                ),
            }
        ),
        "fused_layers": nn.Sequential(
            nn.Linear(25, 150),
            nn.ReLU(),
            nn.Linear(150, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
        ),
        "attention_reduction_ratio": 2,
    },
    "EdgeCorrGNN": {
        "graph_conv_layers": nn.Sequential(
            GCNConv(23, 50),
            GCNConv(50, 100),
            GCNConv(100, 130),
        ),
        "dropout_prob": 0.4,
    },
    "EdgeCorrGraphMaker": {"threshold": 0.6},
    "AttentionWeightedGNN": {
        "graph_conv_layers": nn.Sequential(
            ChebConv(15, 50, K=3),
            ChebConv(50, 100, K=3),
            ChebConv(100, 130, K=3),
        ),
        "dropout_prob": 0.4,
    },
    "AttentionWeightedGraphMaker": {
        "early_stop_callback": EarlyStopping(monitor="val_loss", ),
        "edge_probability_threshold": 80,
        "attention_MLP_test_size": 0.3,
        "AttentionWeightingMLPInstance.weighting_layers": nn.ModuleDict({
            "Layer 1": nn.Sequential(nn.Linear(25, 100),
                                     nn.ReLU()),
            "Layer 2": nn.Sequential(nn.Linear(100, 75),
                                     nn.ReLU()),
            "Layer 3": nn.Sequential(nn.Linear(75, 75),
                                     nn.ReLU()),
            "Layer 4": nn.Sequential(nn.Linear(75, 100),
                                     nn.ReLU()),
            "Layer 5": nn.Sequential(nn.Linear(100, 25),
                                     nn.ReLU()),
        })
    },
}


# Tests for adding modifications and evaluating on new data

@pytest.mark.filterwarnings("ignore:.*does not have many workers*.", )
@pytest.mark.filterwarnings("ignore:.*The number of training batches*.")
@pytest.mark.filterwarnings("ignore:.*No positive samples in targets,*.")
@pytest.mark.filterwarnings("ignore:.*No negative samples in targets,*.")
@pytest.mark.filterwarnings("ignore:.*available but not used*.")
@pytest.mark.filterwarnings("ignore:.*Checkpoint directory.*exists and is not empty*.")
@pytest.mark.filterwarnings("ignore:.*distutils Version classes are deprecated*.")
def test_train_and_test(create_test_files_more_features, tmp_path):
    model_conditions = {"fusion_type": "all"}
    fusion_models = import_chosen_fusion_models(model_conditions, skip_models=["MCVAE_tab"])

    tabular1_csv = create_test_files_more_features["tabular1_csv"]
    tabular2_csv = create_test_files_more_features["tabular2_csv"]
    image_torch_file_2d = create_test_files_more_features["image_torch_file_2d"]

    tabular1_csv_test = create_test_files_more_features["tabular1_csv"]
    tabular2_csv_test = create_test_files_more_features["tabular2_csv"]
    image_torch_file_2d_test = create_test_files_more_features["image_torch_file_2d"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    loss_log_dir = tmp_path / f"loss_log_dir_{timestamp}"
    loss_log_dir.mkdir()

    local_fig_path = tmp_path / f"local_fig_path_{timestamp}"
    local_fig_path.mkdir()
    loss_fig_path = local_fig_path / "losses"
    loss_fig_path.mkdir()

    checkpoint_dir = tmp_path / f"checkpoint_dir_{timestamp}"
    checkpoint_dir.mkdir()

    params = {
        "test_size": 0.2,
        "prediction_task": "binary",
        "multiclass_dimensions": None,
        "kfold": False,
        "wandb_logging": False,
    }

    data_paths = {
        "tabular1": tabular1_csv,
        "tabular2": tabular2_csv,
        "image": image_torch_file_2d,
    }

    test_data_paths = {
        "tabular1": tabular1_csv_test,
        "tabular2": tabular2_csv_test,
        "image": image_torch_file_2d_test,
    }

    output_paths = {
        "checkpoints": str(checkpoint_dir),
        "figures": str(local_fig_path),
        "losses": str(loss_log_dir),
    }

    for model in fusion_models:
        dm = prepare_fusion_data(fusion_model=model,
                                 data_paths=data_paths,
                                 output_paths=output_paths,
                                 params=params,
                                 layer_mods=layer_mods,
                                 max_epochs=2,
                                 **params
                                 )

        single_model_list = train_and_save_models(
            data_module=dm,
            fusion_model=model,
            max_epochs=2,
            enable_checkpointing=True,
            layer_mods=layer_mods,
            wandb_logging=False,
        )

        trained_model = single_model_list[0]

        assert trained_model is not None
        assert trained_model.model is not None

        fig = ConfusionMatrix.from_final_val_data([trained_model])
        assert fig is not None

        if trained_model.model.fusion_type != "graph":
            fig_new_data = ConfusionMatrix.from_new_data([trained_model], output_paths, test_data_paths,
                                                         layer_mods=layer_mods)
            assert fig_new_data is not None

        plt.close("all")


# kfold version

@pytest.mark.filterwarnings("ignore:.*does not have many workers*.", )
@pytest.mark.filterwarnings("ignore:.*The number of training batches*.")
@pytest.mark.filterwarnings("ignore:.*No positive samples in targets,*.")
@pytest.mark.filterwarnings("ignore:.*No negative samples in targets,*.")
@pytest.mark.filterwarnings("ignore:.*available but not used*.")
@pytest.mark.filterwarnings("ignore:.*Checkpoint directory.*exists and is not empty*.")
@pytest.mark.filterwarnings("ignore:.*distutils Version classes are deprecated*.")
def test_kfold(create_test_files_more_features, tmp_path):
    model_conditions = {"class_name": "all"}
    fusion_models = import_chosen_fusion_models(model_conditions, skip_models=["MCVAE_tab"])

    tabular1_csv = create_test_files_more_features["tabular1_csv"]
    tabular2_csv = create_test_files_more_features["tabular2_csv"]
    image_torch_file_2d = create_test_files_more_features["image_torch_file_2d"]

    tabular1_csv_test = create_test_files_more_features["tabular1_csv"]
    tabular2_csv_test = create_test_files_more_features["tabular2_csv"]
    image_torch_file_2d_test = create_test_files_more_features["image_torch_file_2d"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    loss_log_dir = tmp_path / f"loss_log_dir_{timestamp}"
    loss_log_dir.mkdir()

    local_fig_path = tmp_path / f"local_fig_path_{timestamp}"
    local_fig_path.mkdir()
    loss_fig_path = local_fig_path / "losses"
    loss_fig_path.mkdir()

    checkpoint_dir = tmp_path / f"checkpoint_dir_{timestamp}"
    checkpoint_dir.mkdir()

    params = {
        "test_size": 0.2,
        "prediction_task": "binary",
        "multiclass_dimensions": None,
        "kfold": True,
        "num_folds": 3,
        "wandb_logging": False,
    }

    data_paths = {
        "tabular1": tabular1_csv,
        "tabular2": tabular2_csv,
        "image": image_torch_file_2d,
    }

    test_data_paths = {
        "tabular1": tabular1_csv_test,
        "tabular2": tabular2_csv_test,
        "image": image_torch_file_2d_test,
    }

    output_paths = {
        "checkpoints": str(checkpoint_dir),
        "figures": str(local_fig_path),
        "losses": str(loss_log_dir),
    }

    for model in fusion_models:

        dm = prepare_fusion_data(fusion_model=model,
                                 data_paths=data_paths,
                                 output_paths=output_paths,
                                 params=params,
                                 layer_mods=layer_mods,
                                 max_epochs=2,
                                 **params
                                 )

        single_model_list = train_and_save_models(
            data_module=dm,
            fusion_model=model,
            max_epochs=2,
            enable_checkpointing=True,
            layer_mods=layer_mods,
            wandb_logging=False,
        )

        # trained_model = list(single_model_dict.values())[0]

        assert len(single_model_list) == 3
        assert single_model_list[0] is not None
        assert single_model_list[0].model is not None

        fig = ConfusionMatrix.from_final_val_data(single_model_list)
        assert fig is not None

        if single_model_list[0].model.fusion_type != "graph":
            fig_new_data = ConfusionMatrix.from_new_data(single_model_list, output_paths, test_data_paths,
                                                         layer_mods=layer_mods)
            assert fig_new_data is not None

        plt.close("all")
