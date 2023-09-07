import pytest
from unittest.mock import patch
from fusionlibrary.datamodules import CustomDataModule, LoadDatasets, CustomDataset
import shutil
import torch
import pandas as pd
import unittest.mock as mock


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
    image_data_2d = torch.randn(10, 3, 32, 32)
    image_torch_file_2d = tmp_dir / "image_data_2d.pt"
    torch.save(image_data_2d, image_torch_file_2d)

    yield {
        "tabular1_csv": tabular1_csv,
        "tabular2_csv": tabular2_csv,
        "image_torch_file_2d": image_torch_file_2d,
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
    modality_type = "tabular1"

    # Initialize the CustomDataModule
    datamodule = CustomDataModule(params, modality_type, sources, batch_size)
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
    modality_type = "tab_img"

    # Initialize the CustomDataModule
    datamodule = CustomDataModule(params, modality_type, sources, batch_size)
    datamodule.prepare_data()
    datamodule.setup()

    val_dataloader = datamodule.val_dataloader()

    assert val_dataloader.dataset == datamodule.test_dataset
    assert val_dataloader.batch_size == batch_size
    assert type(val_dataloader.sampler) is torch.utils.data.sampler.SequentialSampler


@patch("torch.utils.data.DataLoader", MockedDataLoader)
def test_setup_calls_subspace_method(create_test_files):
    with mock.patch(
        "fusionlibrary.fusion_models.denoise_tab_img_maps.denoising_autoencoder_subspace_method"
    ) as mock_subspace_method:
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
        modality_type = "tab_img"

        # Initialize the CustomDataModule
        datamodule = CustomDataModule(
            params,
            modality_type,
            sources,
            batch_size,
            subspace_method=mock_subspace_method,
        )
        datamodule.prepare_data()
        datamodule.setup()

        mock_subspace_method.assert_called_once_with(datamodule, datamodule.max_epochs)


# Run pytest
if __name__ == "__main__":
    pytest.main()
