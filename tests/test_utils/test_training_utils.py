import pytest
import torch
from fusilli.utils.training_utils import (
    get_file_suffix_from_dict,
    set_checkpoint_name,
    get_checkpoint_filenames_for_subspace_models,
    get_checkpoint_filename_for_trained_fusion_model,
    init_trainer,
    get_final_val_metrics,
)
from unittest.mock import Mock
import wandb
import os
import tempfile
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (EarlyStopping, ModelCheckpoint)
from lightning.pytorch import Trainer
from torchmetrics import Accuracy, R2Score
import numpy as np


# get_file_suffix_from_dict tests
def test_get_file_suffix_from_dict_with_data():
    extra_log_string_dict = {"param1": "value1", "param2": 42}
    extra_name_string, extra_tags = get_file_suffix_from_dict(extra_log_string_dict)
    assert extra_name_string == "_param1_value1_param2_42"
    assert extra_tags == ["param1_value1", "param2_42"]


def test_get_file_suffix_from_dict_with_empty_dict():
    extra_log_string_dict = {}
    extra_name_string, extra_tags = get_file_suffix_from_dict(extra_log_string_dict)
    assert extra_name_string == ""
    assert extra_tags == []


def test_get_file_suffix_from_dict_with_none():
    extra_log_string_dict = None
    extra_name_string, extra_tags = get_file_suffix_from_dict(extra_log_string_dict)
    assert extra_name_string == ""
    assert extra_tags == []


# test set_checkpoint_name
def test_set_checkpoint_name_with_fold():
    class SomeFusionModelClass:
        pass

    fusion_model = SomeFusionModelClass
    fold = 1
    extra_log_string_dict = {"param1": "value1", "param2": 42}

    checkpoint_name = set_checkpoint_name(
        fusion_model, fold, extra_log_string_dict
    )

    assert (
            checkpoint_name
            == "SomeFusionModelClass_fold_1_param1_value1_param2_42_{epoch:02d}"
    )


def test_set_checkpoint_name_without_fold():
    class SomeFusionModelClass:
        pass

    fusion_model = SomeFusionModelClass
    fold = None
    extra_log_string_dict = {"param1": "value1", "param2": 42}

    checkpoint_name = set_checkpoint_name(
        fusion_model, fold, extra_log_string_dict
    )

    assert checkpoint_name == "SomeFusionModelClass_param1_value1_param2_42_{epoch:02d}"


def test_set_checkpoint_name_without_extra_log_string_dict():
    class SomeFusionModelClass:
        pass

    fusion_model = SomeFusionModelClass
    fold = 2
    extra_log_string_dict = None

    checkpoint_name = set_checkpoint_name(
        fusion_model, fold, extra_log_string_dict
    )

    assert checkpoint_name == "SomeFusionModelClass_fold_2_{epoch:02d}"


# test get_checkpoint_filenames_for_subspace_models


def test_get_checkpoint_filenames_for_subspace_models_with_fold():
    class SubspaceModel1:
        pass

    class SubspaceModel2:
        pass

    class SubspaceMethod:
        subspace_models = [SubspaceModel1, SubspaceModel2]

        def __init__(self, datamodule):
            self.datamodule = datamodule

    class SomeFusionModelClass:
        modality_type = "modality_type"

    datamodule = Mock(fusion_model=SomeFusionModelClass(), extra_log_string_dict=None)
    subspace_method = SubspaceMethod(datamodule)
    k = 3

    checkpoint_filenames = get_checkpoint_filenames_for_subspace_models(
        subspace_method, k
    )

    expected_filenames = [
        "subspace_SomeFusionModelClass_SubspaceModel1_fold_3",
        "subspace_SomeFusionModelClass_SubspaceModel2_fold_3",
    ]

    assert checkpoint_filenames == expected_filenames


def test_get_checkpoint_filenames_for_subspace_models_without_fold():
    class SubspaceModel1:
        pass

    class SubspaceModel2:
        pass

    class SubspaceMethod:
        subspace_models = [SubspaceModel1, SubspaceModel2]

        def __init__(self, datamodule):
            self.datamodule = datamodule

    class SomeFusionModelClass:
        modality_type = "modality_type"

    datamodule = Mock(
        fusion_model=SomeFusionModelClass(), extra_log_string_dict={"key": "value"}
    )
    subspace_method = SubspaceMethod(datamodule)

    k = None

    checkpoint_filenames = get_checkpoint_filenames_for_subspace_models(
        subspace_method, k)

    expected_filenames = [
        "subspace_SomeFusionModelClass_SubspaceModel1_key_value",
        "subspace_SomeFusionModelClass_SubspaceModel2_key_value",
    ]

    assert checkpoint_filenames == expected_filenames


# get_checkpoint_filenames_for_trained_fusion_model tests


@pytest.fixture
def params(tmpdir):
    return {
        "checkpoint_dir": str(tmpdir),
    }


@pytest.fixture
def model():
    class MockFusionModel:
        def __init__(self):
            self.__class__.__name__ = "FusionModelClass"

    class MockBaseModel:
        def __init__(self):
            self.model = MockFusionModel()

    return MockBaseModel()


def test_get_checkpoint_filename_for_trained_fusion_model_with_fold(params, model):
    checkpoint_file_suffix = "_suffix"
    fold = 3

    # Create a mock checkpoint file in the directory
    mock_checkpoint_filename = "FusionModelClass_fold_3_suffix"
    mock_checkpoint_path = os.path.join(
        params["checkpoint_dir"], mock_checkpoint_filename
    )
    open(mock_checkpoint_path, "w").close()

    checkpoint_filename = get_checkpoint_filename_for_trained_fusion_model(
        params["checkpoint_dir"], model, checkpoint_file_suffix, fold
    )

    assert checkpoint_filename == mock_checkpoint_path

    # Clean up the mock checkpoint file
    os.remove(mock_checkpoint_path)


def test_get_checkpoint_filename_for_trained_fusion_model_without_fold(params, model):
    checkpoint_file_suffix = "_suffix"

    # Create a mock checkpoint file in the directory
    mock_checkpoint_filename = "FusionModelClass_suffix"
    mock_checkpoint_path = os.path.join(
        params["checkpoint_dir"], mock_checkpoint_filename
    )
    open(mock_checkpoint_path, "w").close()

    checkpoint_filename = get_checkpoint_filename_for_trained_fusion_model(
        params["checkpoint_dir"], model, checkpoint_file_suffix
    )

    assert checkpoint_filename == mock_checkpoint_path

    # Clean up the mock checkpoint file
    os.remove(mock_checkpoint_path)


def test_get_checkpoint_filename_for_trained_fusion_model_not_found(params, model):
    checkpoint_file_suffix = "_suffix"

    # Attempt to get a checkpoint filename when no matching file exists
    with pytest.raises(
            ValueError, match=r"Could not find checkpoint file with name .*"
    ):
        get_checkpoint_filename_for_trained_fusion_model(
            params['checkpoint_dir'], model, checkpoint_file_suffix
        )


def test_get_checkpoint_filename_for_trained_fusion_model_multiple_files(params, model):
    checkpoint_file_suffix = "_suffix"

    # Create multiple mock checkpoint files with the same prefix
    mock_checkpoint_filename_1 = "FusionModelClass_suffix"
    mock_checkpoint_path_1 = os.path.join(
        params["checkpoint_dir"], mock_checkpoint_filename_1
    )
    open(mock_checkpoint_path_1, "w").close()

    mock_checkpoint_filename_2 = "FusionModelClass_suffix_2"
    mock_checkpoint_path_2 = os.path.join(
        params["checkpoint_dir"], mock_checkpoint_filename_2
    )
    open(mock_checkpoint_path_2, "w").close()

    # Attempt to get a checkpoint filename when multiple matching files exist
    with pytest.raises(
            ValueError, match=r"Found multiple checkpoint files with name .*"
    ):
        get_checkpoint_filename_for_trained_fusion_model(
            params["checkpoint_dir"], model, checkpoint_file_suffix
        )

    # Clean up the mock checkpoint files
    os.remove(mock_checkpoint_path_1)
    os.remove(mock_checkpoint_path_2)


# from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
# from pytorch_lightning import Trainer
# from lightning.pytorch.callbacks.early_stopping import EarlyStopping
# from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
# from lightning.pytorch.trainer.trainer import Trainer


@pytest.fixture
def mock_logger():
    # Create a mock logger object for testing
    return Mock()


def test_init_trainer_default(mock_logger):
    # Test initializing trainer with default parameters
    trainer = init_trainer(mock_logger, output_paths={}, )
    assert trainer is not None
    assert isinstance(trainer, Trainer)
    assert trainer.max_epochs == 1000
    assert trainer.num_sanity_val_steps == 0
    assert trainer.logger == mock_logger
    assert isinstance(trainer.callbacks[0], EarlyStopping)
    assert trainer.checkpoint_callback is not None


@pytest.mark.filterwarnings("ignore:.*GPU available but not used*.", )
def test_init_trainer_custom_early_stopping(mock_logger):
    # Test initializing trainer with a custom early stopping callback
    # custom_early_stopping = Mock()
    custom_early_stopping = EarlyStopping(monitor="val_loss",
                                          patience=3,
                                          verbose=True,
                                          mode="max", )
    trainer = init_trainer(
        mock_logger, output_paths={}, own_early_stopping_callback=custom_early_stopping
    )
    assert trainer is not None
    assert isinstance(trainer, Trainer)
    assert trainer.max_epochs == 1000
    assert trainer.num_sanity_val_steps == 0
    assert trainer.logger == mock_logger

    # check that the custom early stopping callback is the first callback
    assert trainer.early_stopping_callback is not None
    assert isinstance(trainer.callbacks[0], EarlyStopping)
    assert trainer.callbacks[0] == custom_early_stopping
    for key in custom_early_stopping.__dict__:
        assert custom_early_stopping.__dict__[key] == trainer.early_stopping_callback.__dict__[key]

    assert trainer.checkpoint_callback is not None


def test_init_trainer_with_checkpointing(mock_logger):
    # Test initializing trainer with checkpointing enabled

    trainer = init_trainer(
        mock_logger,
        output_paths={"checkpoints": tempfile.mkdtemp()},
        enable_checkpointing=True,
        checkpoint_filename="model_checkpoint.pth",
    )
    assert trainer is not None
    assert isinstance(trainer, Trainer)
    assert trainer.max_epochs == 1000
    assert trainer.num_sanity_val_steps == 0
    assert trainer.logger == mock_logger
    assert isinstance(trainer.callbacks[0], EarlyStopping)
    assert trainer.checkpoint_callback is not None
    assert trainer.checkpoint_callback.filename == "model_checkpoint.pth"


def test_init_trainer_with_accelerator_and_devices(mock_logger):
    # Test initializing trainer with custom accelerator and devices

    params = {"accelerator": "cpu", "devices": 3}
    trainer = init_trainer(mock_logger, output_paths={}, training_modifications={"accelerator": "cpu", "devices": 3})

    assert trainer is not None
    assert isinstance(trainer, Trainer)
    assert trainer.max_epochs == 1000
    assert trainer.num_sanity_val_steps == 0
    assert trainer.logger == mock_logger
    assert str(trainer._accelerator_connector._accelerator_flag) == "cpu"
    if str(trainer._accelerator_connector._accelerator_flag) == "cpu":
        assert trainer._accelerator_connector._devices_flag == 3
    assert isinstance(trainer.callbacks[0], EarlyStopping)


def test_init_trainer_without_checkpointing(mock_logger):
    # Test initializing trainer with checkpointing disabled
    trainer = init_trainer(mock_logger, output_paths={}, enable_checkpointing=False)
    assert trainer is not None
    assert isinstance(trainer, Trainer)
    assert trainer.max_epochs == 1000
    assert trainer.num_sanity_val_steps == 0
    assert trainer.logger == mock_logger
    assert isinstance(trainer.callbacks[0], EarlyStopping)


# from pytorch_lightning import Trainer
# from lightning.pytorch.trainer import Trainer
# from torchmetrics import Accuracy, R2Score
# import numpy as np


# Define a dummy LightningModule class for testing
class DummyLightningModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.metric1_val = Accuracy(task="binary")
        self.metric2_val = R2Score()
        self.metric_names_list = ["Accuracy", "R2"]


# Test case 1: Check if the function returns the correct metrics
def test_get_final_val_metrics():
    # Create a dummy LightningModule and a Trainer
    model = DummyLightningModule()

    class MockTrainer:
        def __init__(self, model):
            self.callback_metrics = {}
            self.model = model

    trainer = MockTrainer(model)

    # Set some dummy metrics in the Trainer
    trainer.callback_metrics = {
        "Accuracy_val": torch.tensor(0.5),  # Example value
        "R2_val": torch.tensor(0.8),  # Example value
    }

    # Get the final validation metrics
    metric1, metric2 = get_final_val_metrics(trainer)

    # Check if the returned metrics match the expected values
    assert np.isclose(metric1, 0.5)
    assert np.isclose(metric2, 0.8)


# Check if the function handles empty Trainer metrics
def test_get_final_val_metrics_empty_metrics():
    # Create a dummy LightningModule and a Trainer
    model = DummyLightningModule()

    class MockTrainer:
        def __init__(self, model):
            self.callback_metrics = {}
            self.model = model

    trainer = MockTrainer(model)

    # Get the final validation metrics
    with pytest.raises(ValueError, match=r"trainer.callback_metrics is empty."):
        get_final_val_metrics(trainer)


# raises error with wrong metric names
def test_get_final_val_metrics_wrong_metric_names():
    # Create a dummy LightningModule and a Trainer
    model = DummyLightningModule()

    class MockTrainer:
        def __init__(self, model):
            self.callback_metrics = {}
            self.model = model

    trainer = MockTrainer(model)

    # Set some dummy metrics in the Trainer
    trainer.callback_metrics = {
        "Accuracy_val_wrong": torch.tensor(0.5),  # Example value
        "R2_val": torch.tensor(0.8),  # Example value
    }

    # Get the final validation metrics
    with pytest.raises(
            ValueError,
            match=r"not in trainer.callback_metrics.keys()",
    ):
        get_final_val_metrics(trainer)
