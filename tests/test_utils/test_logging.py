import os
from unittest.mock import patch, Mock
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from fusilli.utils.training_utils import set_logger
import wandb

# API key for throwaway WandB account
wandb.login(key="1a4c82a7e7fa7a8f24cccca821c17d3e4b065835")


def test_set_logger_with_kfold_flag_true():
    wandb.finish()
    params = {
        "log": True,
        "kfold_flag": True,
        "project_name": "2023-10-05",
    }
    fold = 1

    class SomeFusionModelClass:
        modality_type = "modality_type"
        fusion_type = "fusion_type"

    fusion_model = SomeFusionModelClass()

    extra_log_string_dict = {"param1": "value1", "param2": 42}

    with patch("os.getcwd", return_value="/mocked/path"):
        logger = set_logger(fold=fold,
                            project_name=params["project_name"],
                            output_paths={},
                            fusion_model=fusion_model,
                            extra_log_string_dict=extra_log_string_dict,
                            wandb_logging=True)

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
        "project_name": "2023-10-05",
    }
    fold = None

    class SomeFusionModelClass:
        modality_type = "modality_type"
        fusion_type = "fusion_type"

    fusion_model = SomeFusionModelClass()

    extra_log_string_dict = {"param1": "value1", "param2": 42}

    with patch("os.getcwd", return_value="/mocked/path"):
        logger = set_logger(fold=fold,
                            project_name=params["project_name"],
                            output_paths={},
                            fusion_model=fusion_model,
                            extra_log_string_dict=extra_log_string_dict,
                            wandb_logging=True)

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
    fold = 1
    fusion_model = Mock()
    extra_log_string_dict = {"param1": "value1", "param2": 42}

    logger = set_logger(fold=fold,
                        project_name=None,
                        output_paths={"losses": "loss_log_dir"},
                        fusion_model=fusion_model,
                        extra_log_string_dict=extra_log_string_dict,
                        wandb_logging=False)

    assert isinstance(logger, CSVLogger)
    assert logger.version == "Mock_fold_1_param1_value1_param2_42"
    assert logger.save_dir == "loss_log_dir"
