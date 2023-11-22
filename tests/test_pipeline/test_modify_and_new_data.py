"""
These tests go through the whole pipeline and check that we can modify the model structure
and the subspace model structure and the from_new_data methods will still work without errors.
"""

import pytest
from fusilli.data import get_data_module
from fusilli.train import train_and_save_models
from fusilli.utils.model_chooser import import_chosen_fusion_models
from fusilli.eval import ConfusionMatrix
from ..test_data.test_TrainTestDataModule import create_test_files
import torch.nn as nn
from datetime import datetime
import os


@pytest.mark.filterwarnings("ignore:.*does not have many workers*.", )
@pytest.mark.filterwarnings("ignore:.*The number of training batches*.")
@pytest.mark.filterwarnings("ignore:.*No positive samples in targets,*.")
@pytest.mark.filterwarnings("ignore:.*available but not used*.")
@pytest.mark.filterwarnings("ignore:.*Checkpoint directory.*exists and is not empty*.")
@pytest.mark.filterwarnings("ignore:.*distutils Version classes are deprecated*.")
def test_train_and_test(create_test_files, tmp_path):
    model_conditions = {"class_name": ["DAETabImgMaps"]}
    model = import_chosen_fusion_models(model_conditions, skip_models=["MCVAE_tab"])[0]

    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    image_torch_file_2d = create_test_files["image_torch_file_2d"]

    tabular1_csv_test = create_test_files["tabular1_csv"]
    tabular2_csv_test = create_test_files["tabular2_csv"]
    image_torch_file_2d_test = create_test_files["image_torch_file_2d"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    loss_fig_path = tmp_path / f"loss_fig_path_{timestamp}"
    loss_fig_path.mkdir()

    loss_log_dir = tmp_path / f"loss_log_dir_{timestamp}"
    loss_log_dir.mkdir()

    local_fig_path = tmp_path / f"local_fig_path_{timestamp}"
    local_fig_path.mkdir()

    checkpoint_dir = tmp_path / f"checkpoint_dir_{timestamp}"
    checkpoint_dir.mkdir()

    params = {
        "test_size": 0.2,
        "pred_type": "binary",
        "multiclass_dims": None,
        "kfold_flag": False,
        "tabular1_source": tabular1_csv,
        "tabular2_source": tabular2_csv,
        "img_source": image_torch_file_2d,
        "tabular1_source_test": tabular1_csv_test,
        "tabular2_source_test": tabular2_csv_test,
        "img_source_test": image_torch_file_2d_test,
        "log": False,
        "loss_fig_path": str(loss_fig_path),
        "loss_log_dir": str(loss_log_dir),
        "local_fig_path": str(local_fig_path),
        "checkpoint_dir": str(checkpoint_dir),

    }

    layer_mods = {
        "denoising_autoencoder_subspace_method": {
            "autoencoder.latent_dim": 55,
        },
        "DAETabImgMaps": {
            "fusion_layers": nn.Sequential(
                nn.Linear(750, 480),
                nn.ReLU(),
                nn.Linear(480, 220),
                nn.ReLU(),
                nn.Linear(220, 88),
            )
        }
    }

    dm = get_data_module(fusion_model=model,
                         params=params,
                         layer_mods=layer_mods,
                         max_epochs=20,
                         )

    single_model_list = train_and_save_models(
        data_module=dm,
        params=params,
        fusion_model=model,
        max_epochs=20,
        layer_mods=layer_mods,
    )

    trained_model = single_model_list[0]

    assert trained_model is not None
    assert trained_model.model is not None

    fig = ConfusionMatrix.from_final_val_data([trained_model])
    assert fig is not None

    fig_new_data = ConfusionMatrix.from_new_data([trained_model], params, "_test",
                                                 layer_mods=layer_mods)
    assert fig_new_data is not None


# kfold version

@pytest.mark.filterwarnings("ignore:.*does not have many workers*.", )
@pytest.mark.filterwarnings("ignore:.*The number of training batches*.")
@pytest.mark.filterwarnings("ignore:.*No positive samples in targets,*.")
@pytest.mark.filterwarnings("ignore:.*available but not used*.")
@pytest.mark.filterwarnings("ignore:.*Checkpoint directory.*exists and is not empty*.")
@pytest.mark.filterwarnings("ignore:.*distutils Version classes are deprecated*.")
def test_kfold(create_test_files, tmp_path):
    model_conditions = {"class_name": ["DAETabImgMaps"]}
    model = import_chosen_fusion_models(model_conditions, skip_models=["MCVAE_tab"])[0]

    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    image_torch_file_2d = create_test_files["image_torch_file_2d"]

    tabular1_csv_test = create_test_files["tabular1_csv"]
    tabular2_csv_test = create_test_files["tabular2_csv"]
    image_torch_file_2d_test = create_test_files["image_torch_file_2d"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    loss_fig_path = tmp_path / f"loss_fig_path_{timestamp}"
    loss_fig_path.mkdir()

    loss_log_dir = tmp_path / f"loss_log_dir_{timestamp}"
    loss_log_dir.mkdir()

    local_fig_path = tmp_path / f"local_fig_path_{timestamp}"
    local_fig_path.mkdir()

    checkpoint_dir = tmp_path / f"checkpoint_dir_{timestamp}"
    checkpoint_dir.mkdir()

    params = {
        "test_size": 0.2,
        "pred_type": "binary",
        "multiclass_dims": None,
        "kfold_flag": True,
        "num_k": 3,
        "tabular1_source": tabular1_csv,
        "tabular2_source": tabular2_csv,
        "img_source": image_torch_file_2d,
        "tabular1_source_test": tabular1_csv_test,
        "tabular2_source_test": tabular2_csv_test,
        "img_source_test": image_torch_file_2d_test,
        "log": False,
        "loss_fig_path": str(loss_fig_path),
        "loss_log_dir": str(loss_log_dir),
        "local_fig_path": str(local_fig_path),
        "checkpoint_dir": str(checkpoint_dir),

    }

    # altering both subspace method and fusion model
    layer_mods = {
        "denoising_autoencoder_subspace_method": {
            "autoencoder.latent_dim": 55,
        },
        "DAETabImgMaps": {
            "fusion_layers": nn.Sequential(
                nn.Linear(750, 480),
                nn.ReLU(),
                nn.Linear(480, 220),
                nn.ReLU(),
                nn.Linear(220, 88),
            )
        }
    }

    dm = get_data_module(fusion_model=model,
                         params=params,
                         layer_mods=layer_mods,
                         max_epochs=20,
                         )

    single_model_list = train_and_save_models(
        data_module=dm,
        params=params,
        fusion_model=model,
        max_epochs=20,
        layer_mods=layer_mods,
    )

    # trained_model = list(single_model_dict.values())[0]

    assert len(single_model_list) == 3
    assert single_model_list[0] is not None
    assert single_model_list[0].model is not None

    fig = ConfusionMatrix.from_final_val_data(single_model_list)
    assert fig is not None

    fig_new_data = ConfusionMatrix.from_new_data(single_model_list, params, "_test",
                                                 layer_mods=layer_mods)
    assert fig_new_data is not None
