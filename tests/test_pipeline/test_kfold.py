import pytest
from fusilli.data import prepare_fusion_data
from fusilli.train import train_and_save_models
from fusilli.utils.model_chooser import import_chosen_fusion_models
from fusilli.eval import ConfusionMatrix
from ..test_data.test_TrainTestDataModule import create_test_files
import matplotlib.pyplot as plt


@pytest.mark.filterwarnings("ignore:.*does not have many workers*.", )
@pytest.mark.filterwarnings("ignore:.*The number of training batches*.")
@pytest.mark.filterwarnings("ignore:.*No positive samples in targets,*.")
@pytest.mark.filterwarnings("ignore:.*No negative samples in targets,*.")
@pytest.mark.filterwarnings("ignore:.*exists and is not empty*.")
def test_5fold_cv(create_test_files, tmp_path):
    model_conditions = {"modality_type": "all", }

    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    image_torch_file_2d = create_test_files["image_torch_file_2d"]

    loss_fig_path = tmp_path / "loss_fig_path"
    loss_fig_path.mkdir()

    loss_log_dir = tmp_path / "loss_log_dir"
    loss_log_dir.mkdir()

    local_fig_path = tmp_path / "local_fig_path"
    local_fig_path.mkdir()
    loss_fig_path = local_fig_path / "losses"
    loss_fig_path.mkdir()

    checkpoint_dir = tmp_path / "checkpoint_dir"
    checkpoint_dir.mkdir()

    modifications = {
        "AttentionAndSelfActivation": {"attention_reduction_ratio": 2}
    }

    params = {
        # "test_size": 0.2,
        "prediction_task": "binary",
        "multiclass_dimensions": None,
        "kfold": True,
        "num_folds": 5,
        "wandb_logging": False,
    }

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

    new_metrics = ["accuracy", "precision", "recall", "f1", "auroc", "auprc", "balanced_accuracy"]

    fusion_models = import_chosen_fusion_models(model_conditions, skip_models=["MCVAE_tab"])

    for model in fusion_models:
        print("model", model)
        dm = prepare_fusion_data(fusion_model=model,
                                 data_paths=data_paths,
                                 output_paths=output_paths,
                                 layer_mods=modifications,
                                 max_epochs=2,
                                 **params)

        single_model_list = train_and_save_models(
            data_module=dm,
            fusion_model=model,
            max_epochs=2,
            enable_checkpointing=False,
            layer_mods=modifications,
            wandb_logging=False,
            metrics_list=new_metrics,
        )

        assert single_model_list is not None
        assert len(single_model_list) == 5

        fig = ConfusionMatrix.from_final_val_data(single_model_list)
        assert fig is not None

        plt.close("all")
