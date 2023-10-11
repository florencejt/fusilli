import pytest
from unittest.mock import patch
from fusilli.data import TrainTestDataModule, LoadDatasets, CustomDataset
import shutil
import torch
import pandas as pd
from unittest.mock import patch, Mock


@pytest.fixture(scope="module")
def create_test_files(tmp_path_factory):
    # Create a temporary directory
    tmp_dir = tmp_path_factory.mktemp("test_data")

    # Create sample CSV files with different index and label column names
    tabular1_csv = tmp_dir / "tabular1.csv"
    tabular1_data = pd.DataFrame(
        {
            "study_id": range(10),  # Different index column name
            "feature1": [1.0] * 10,
            "feature2": [2.0] * 10,
            "pred_label": [0] * 10,  # Different label column name
        }
    )
    tabular1_data.to_csv(tabular1_csv, index=False)

    tabular2_csv = tmp_dir / "tabular2.csv"
    tabular2_data = pd.DataFrame(
        {
            "study_id": range(10),
            "feature3": [3.0] * 10,
            "feature4": [4.0] * 10,
            "pred_label": [1] * 10,
        }
    )
    tabular2_data.to_csv(tabular2_csv, index=False)

    # Create a sample Torch file for image data
    image_data_2d = torch.randn(10, 1, 100, 100)
    image_torch_file_2d = tmp_dir / "image_data_2d.pt"
    torch.save(image_data_2d, image_torch_file_2d)

    # Create a sample Torch file for image data
    image_data_3d = torch.randn(10, 1, 100, 100, 100)
    image_torch_file_3d = tmp_dir / "image_data_3d.pt"
    torch.save(image_data_3d, image_torch_file_3d)

    yield {
        "tabular1_csv": tabular1_csv,
        "tabular2_csv": tabular2_csv,
        "image_torch_file_2d": image_torch_file_2d,
        "image_torch_file_3d": image_torch_file_3d,
    }

    # Clean up temporary files and directories
    shutil.rmtree(tmp_dir)


# Mocked class for DataLoader
class MockedDataLoader:
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers


# Test case for training dataloader
@patch("torch.utils.data.DataLoader", MockedDataLoader)
def test_train_dataloader(create_test_files):
    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    image_torch_file_2d = create_test_files["image_torch_file_2d"]

    params = {
        "test_size": 0.2,
        "pred_type": "binary",
        "multiclass_dims": None,
    }
    sources = [tabular1_csv, tabular2_csv, image_torch_file_2d]
    batch_size = 8

    example_fusion_model = Mock()
    example_fusion_model.modality_type = "tabular1"

    # Initialize the TrainTestDataModule
    datamodule = TrainTestDataModule(params, example_fusion_model, sources, batch_size)
    datamodule.prepare_data()
    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()

    assert train_dataloader.dataset == datamodule.train_dataset
    assert train_dataloader.batch_size == batch_size
    assert type(train_dataloader.sampler) is torch.utils.data.sampler.RandomSampler


# Test case for validation dataloader
@patch("torch.utils.data.DataLoader", MockedDataLoader)
def test_val_dataloader(create_test_files):
    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    image_torch_file_2d = create_test_files["image_torch_file_2d"]

    params = {
        "test_size": 0.2,
        "pred_type": "binary",
        "multiclass_dims": None,
    }

    sources = [tabular1_csv, tabular2_csv, image_torch_file_2d]
    batch_size = 23

    example_fusion_model = Mock()
    example_fusion_model.modality_type = "tab_img"

    # Initialize the TrainTestDataModule
    datamodule = TrainTestDataModule(params, example_fusion_model, sources, batch_size)
    datamodule.prepare_data()
    datamodule.setup()

    val_dataloader = datamodule.val_dataloader()

    assert val_dataloader.dataset == datamodule.test_dataset
    assert val_dataloader.batch_size == batch_size
    assert type(val_dataloader.sampler) is torch.utils.data.sampler.SequentialSampler


class MockSubspaceMethod:
    def train(self, train_dataset, test_dataset):
        # Simulate the behavior of the train method here
        train_latents = torch.Tensor([[0.1, 0.2, 0.3]])
        train_labels = pd.DataFrame([0.3], columns=["pred_label"])
        return train_latents, train_labels

    def convert_to_latent(self, test_dataset):
        return (
            torch.Tensor([[0.1, 0.2, 0.3]]),
            pd.DataFrame([0.3], columns=["pred_label"]),
            [3, None, None],
        )


def test_setup_calls_subspace_method(create_test_files):
    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    image_torch_file_2d = create_test_files["image_torch_file_2d"]

    params = {
        "test_size": 0.2,
        "pred_type": "binary",
        "multiclass_dims": None,
    }

    sources = [tabular1_csv, tabular2_csv, image_torch_file_2d]
    batch_size = 23

    example_fusion_model = Mock()
    example_fusion_model.modality_type = "tab_img"

    with patch(
        "fusilli.fusion_models.denoise_tab_img_maps.denoising_autoencoder_subspace_method",
        return_value=MockSubspaceMethod(),
    ) as mock_subspace_method:
        # Initialize the TrainTestDataModule
        datamodule = TrainTestDataModule(
            params,
            example_fusion_model,
            sources,
            batch_size,
            subspace_method=mock_subspace_method,
        )
        datamodule.prepare_data()
        datamodule.setup()

        # Assert that the subspace_method class was called
        mock_subspace_method.assert_called_once_with(
            datamodule, max_epochs=datamodule.max_epochs, k=None
        )


# Run pytest
if __name__ == "__main__":
    pytest.main()
