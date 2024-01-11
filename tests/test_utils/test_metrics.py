"""
Testing modifying metrics
"""
import pytest
from fusilli.data import prepare_fusion_data
from fusilli.train import train_and_save_models
from fusilli.utils.model_chooser import import_chosen_fusion_models
from fusilli.eval import ConfusionMatrix, RealsVsPreds
from ..test_data.test_TrainTestDataModule import create_test_files
from datetime import datetime

binary_metrics = ["auroc", "accuracy", "recall", "specificity", "precision", "f1", "auprc", "balanced_accuracy"]

regression_metrics = ["r2", "mae", "mse"]


@pytest.mark.filterwarnings("ignore:.*does not have many workers*.", )
@pytest.mark.filterwarnings("ignore:.*The number of training batches*.")
@pytest.mark.filterwarnings("ignore:.*No positive samples in targets,*.")
@pytest.mark.filterwarnings("ignore:.*No negative samples in targets,*.")
@pytest.mark.filterwarnings("ignore:.*exists and is not empty*.")
@pytest.mark.parametrize("metric", binary_metrics)
def test_regression_metrics_for_classification(metric, create_test_files, tmp_path):
    model_conditions = {"class_name": "ConcatTabularData"}
    fusion_models = import_chosen_fusion_models(model_conditions, skip_models=["MCVAE_tab"])

    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    image_torch_file_2d = create_test_files["image_torch_file_2d"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    loss_log_dir = tmp_path / f"loss_log_dir_{timestamp}"
    loss_log_dir.mkdir()

    local_fig_path = tmp_path / f"local_fig_path_{timestamp}"
    local_fig_path.mkdir()
    loss_fig_path = local_fig_path / "losses"
    loss_fig_path.mkdir()

    checkpoint_dir = tmp_path / f"checkpoint_dir_{timestamp}"
    checkpoint_dir.mkdir()

    data_paths = {
        "tabular1": tabular1_csv,
        "tabular2": tabular2_csv,
        "image": image_torch_file_2d,
    }

    output_paths = {
        "checkpoints": str(checkpoint_dir),
        "figures": str(local_fig_path),
        "losses": str(loss_log_dir),
    }

    new_metrics = ["r2", metric]

    dm = prepare_fusion_data(prediction_task="regression",
                             fusion_model=fusion_models[0],
                             data_paths=data_paths,
                             output_paths=output_paths,
                             )

    # raises error
    with pytest.raises(ValueError, match=r"Invalid prediction task for."):
        single_model_list = train_and_save_models(
            data_module=dm,
            fusion_model=fusion_models[0],
            max_epochs=2,
            enable_checkpointing=False,
            wandb_logging=False,
            metrics_list=new_metrics,
        )


# error when trying to use classification metrics for regression
@pytest.mark.filterwarnings("ignore:.*does not have many workers*.", )
@pytest.mark.filterwarnings("ignore:.*The number of training batches*.")
@pytest.mark.filterwarnings("ignore:.*No positive samples in targets,*.")
@pytest.mark.filterwarnings("ignore:.*No negative samples in targets,*.")
@pytest.mark.filterwarnings("ignore:.*exists and is not empty*.")
@pytest.mark.parametrize("metric", regression_metrics)
def test_binary_metrics_for_regression(metric, create_test_files, tmp_path):
    model_conditions = {"class_name": "ConcatTabularData"}
    fusion_models = import_chosen_fusion_models(model_conditions, skip_models=["MCVAE_tab"])

    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    image_torch_file_2d = create_test_files["image_torch_file_2d"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    loss_log_dir = tmp_path / f"loss_log_dir_{timestamp}"
    loss_log_dir.mkdir()

    local_fig_path = tmp_path / f"local_fig_path_{timestamp}"
    local_fig_path.mkdir()
    loss_fig_path = local_fig_path / "losses"
    loss_fig_path.mkdir()

    checkpoint_dir = tmp_path / f"checkpoint_dir_{timestamp}"
    checkpoint_dir.mkdir()

    data_paths = {
        "tabular1": tabular1_csv,
        "tabular2": tabular2_csv,
        "image": image_torch_file_2d,
    }

    output_paths = {
        "checkpoints": str(checkpoint_dir),
        "figures": str(local_fig_path),
        "losses": str(loss_log_dir),
    }

    new_metrics = ["auroc", metric]

    dm = prepare_fusion_data(prediction_task="binary",
                             fusion_model=fusion_models[0],
                             data_paths=data_paths,
                             output_paths=output_paths,
                             )

    # raises error
    with pytest.raises(ValueError, match=r"Invalid prediction task for."):
        single_model_list = train_and_save_models(
            data_module=dm,
            fusion_model=fusion_models[0],
            max_epochs=2,
            enable_checkpointing=False,
            wandb_logging=False,
            metrics_list=new_metrics,
        )


# error when only providing one metric
def test_only_one_metric(create_test_files, tmp_path):
    model_conditions = {"class_name": "ConcatTabularData"}
    fusion_models = import_chosen_fusion_models(model_conditions, skip_models=["MCVAE_tab"])

    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    image_torch_file_2d = create_test_files["image_torch_file_2d"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    loss_log_dir = tmp_path / f"loss_log_dir_{timestamp}"
    loss_log_dir.mkdir()

    local_fig_path = tmp_path / f"local_fig_path_{timestamp}"
    local_fig_path.mkdir()
    loss_fig_path = local_fig_path / "losses"
    loss_fig_path.mkdir()

    checkpoint_dir = tmp_path / f"checkpoint_dir_{timestamp}"
    checkpoint_dir.mkdir()

    data_paths = {
        "tabular1": tabular1_csv,
        "tabular2": tabular2_csv,
        "image": image_torch_file_2d,
    }

    output_paths = {
        "checkpoints": str(checkpoint_dir),
        "figures": str(local_fig_path),
        "losses": str(loss_log_dir),
    }

    new_metrics = ["auroc"]

    dm = prepare_fusion_data(prediction_task="binary",
                             fusion_model=fusion_models[0],
                             data_paths=data_paths,
                             output_paths=output_paths,
                             )

    # raises error
    with pytest.raises(ValueError, match=r"Length of metrics list must be 2 or more."):
        single_model_list = train_and_save_models(
            data_module=dm,
            fusion_model=fusion_models[0],
            max_epochs=2,
            enable_checkpointing=False,
            wandb_logging=False,
            metrics_list=new_metrics,
        )


# error when only providing one metric
def test_unsupported_metric(create_test_files, tmp_path):
    model_conditions = {"class_name": "ConcatTabularData"}
    fusion_models = import_chosen_fusion_models(model_conditions, skip_models=["MCVAE_tab"])

    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    image_torch_file_2d = create_test_files["image_torch_file_2d"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    loss_log_dir = tmp_path / f"loss_log_dir_{timestamp}"
    loss_log_dir.mkdir()

    local_fig_path = tmp_path / f"local_fig_path_{timestamp}"
    local_fig_path.mkdir()
    loss_fig_path = local_fig_path / "losses"
    loss_fig_path.mkdir()

    checkpoint_dir = tmp_path / f"checkpoint_dir_{timestamp}"
    checkpoint_dir.mkdir()

    data_paths = {
        "tabular1": tabular1_csv,
        "tabular2": tabular2_csv,
        "image": image_torch_file_2d,
    }

    output_paths = {
        "checkpoints": str(checkpoint_dir),
        "figures": str(local_fig_path),
        "losses": str(loss_log_dir),
    }

    new_metrics = ["auroc", "unsupported_metric"]

    dm = prepare_fusion_data(prediction_task="binary",
                             fusion_model=fusion_models[0],
                             data_paths=data_paths,
                             output_paths=output_paths,
                             )

    # raises error
    with pytest.raises(ValueError, match=r"Unsupported metric: unsupported_metric."):
        single_model_list = train_and_save_models(
            data_module=dm,
            fusion_model=fusion_models[0],
            max_epochs=2,
            enable_checkpointing=False,
            wandb_logging=False,
            metrics_list=new_metrics,
        )
