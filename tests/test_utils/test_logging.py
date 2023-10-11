import os
from unittest.mock import patch, Mock
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from fusilli.utils.training_utils import set_logger
import wandb


def test_set_logger_with_kfold_flag_true():
    wandb.finish()
    params = {
        "log": True,
        "kfold_flag": True,
        "timestamp": "2023-10-05",
    }
    fold = 1

    class SomeFusionModelClass:
        modality_type = "modality_type"
        fusion_type = "fusion_type"

    fusion_model = SomeFusionModelClass()

    extra_log_string_dict = {"param1": "value1", "param2": 42}

    with patch("os.getcwd", return_value="/mocked/path"):
        logger = set_logger(params, fold, fusion_model, extra_log_string_dict)

    assert isinstance(logger, WandbLogger)
    assert logger.save_dir == "/mocked/path/logs"
    assert logger._project == "2023-10-05"
    assert logger._name == "SomeFusionModelClass_fold_1_param1_value1_param2_42"
    assert logger._log_model
    assert logger._wandb_init
    assert logger.experiment.config["method_name"] == "SomeFusionModelClass"
    assert logger.experiment.config["param1"] == "value1"
    assert logger.experiment.config["param2"] == 42


def test_set_logger_with_kfold_flag_false():
    wandb.finish()
    params = {
        "log": True,
        "kfold_flag": False,
        "timestamp": "2023-10-05",
    }
    fold = None

    class SomeFusionModelClass:
        modality_type = "modality_type"
        fusion_type = "fusion_type"

    fusion_model = SomeFusionModelClass()

    extra_log_string_dict = {"param1": "value1", "param2": 42}

    with patch("os.getcwd", return_value="/mocked/path"):
        logger = set_logger(params, fold, fusion_model, extra_log_string_dict)

    assert isinstance(logger, WandbLogger)
    assert logger.save_dir == "/mocked/path/logs"
    assert logger._project == "2023-10-05"
    assert logger._name == "SomeFusionModelClass_param1_value1_param2_42"
    assert logger._log_model
    assert logger._wandb_init
    assert logger.experiment.config["method_name"] == "SomeFusionModelClass"
    assert logger.experiment.config["param1"] == "value1"
    assert logger.experiment.config["param2"] == 42


def test_set_logger_with_log_false():
    params = {
        "log": False,
        "kfold_flag": True,
        "timestamp": "2023-10-05",
        "loss_log_dir": "loss_log_dir",
    }
    fold = 1
    fusion_model = Mock()
    extra_log_string_dict = {"param1": "value1", "param2": 42}

    logger = set_logger(params, fold, fusion_model, extra_log_string_dict)

    assert isinstance(logger, CSVLogger)
