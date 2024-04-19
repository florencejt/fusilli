import pytest
from fusilli.data import prepare_fusion_data
from fusilli.train import train_and_save_models
from fusilli.utils.model_chooser import import_chosen_fusion_models
from fusilli.eval import ConfusionMatrix
from ..test_data.test_TrainTestDataModule import create_test_files
from datetime import datetime


@pytest.mark.filterwarnings(
    "ignore:.*does not have many workers*.",
)
@pytest.mark.filterwarnings("ignore:.*The number of training batches*.")
@pytest.mark.filterwarnings("ignore:.*No positive samples in targets,*.")
@pytest.mark.filterwarnings("ignore:.*No negative samples in targets,*.")
@pytest.mark.filterwarnings("ignore:.*exists and is not empty*.")
def test_train_and_test(create_test_files, tmp_path):
    model_conditions = {"modality_type": "all"}
    fusion_models = import_chosen_fusion_models(
        model_conditions, skip_models=["MCVAE_tab"]
    )

    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    # third tabular modality
    # TODO make this tabular3_csv in the function
    tabular3_csv = create_test_files["tabular3_csv"]

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
        "tabular3": tabular3_csv,  # TODO add this functionality to fusilli
    }

    output_paths = {
        "checkpoints": str(checkpoint_dir),
        "figures": str(local_fig_path),
        "losses": str(loss_log_dir),
    }

    modifications = {"AttentionAndSelfActivation": {"attention_reduction_ratio": 2}}

    new_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auroc",
        "auprc",
        "balanced_accuracy",
    ]

    for model in fusion_models:

        # TODO check that the model is able to handle 3 tabular modalities
        # - or add a warning and then skip if it's not able to

        dm = prepare_fusion_data(
            fusion_model=model,
            data_paths=data_paths,
            output_paths=output_paths,
            params=params,
            max_epochs=2,
            layer_mods=modifications,
            **params,
        )

        single_model_list = train_and_save_models(
            data_module=dm,
            fusion_model=model,
            max_epochs=2,
            enable_checkpointing=False,
            wandb_logging=False,
            layer_mods=modifications,
            metrics_list=new_metrics,
        )

        trained_model = single_model_list[0]

        assert trained_model is not None
        assert trained_model.model is not None

        fig = ConfusionMatrix.from_final_val_data([trained_model])
        assert fig is not None