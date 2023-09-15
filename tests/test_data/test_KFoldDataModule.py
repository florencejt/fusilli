import pytest
import torch
from fusionlibrary.data import KFoldDataModule
from .test_CustomDataModule import create_test_files, MockSubspaceMethod
from unittest.mock import patch, Mock


@pytest.fixture
def create_kfold_data_module(create_test_files):
    params = {
        "num_k": 5,
        "pred_type": "binary",
        "multiclass_dims": None,
    }

    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    image_torch_file_2d = create_test_files["image_torch_file_2d"]

    sources = [tabular1_csv, tabular2_csv, image_torch_file_2d]
    batch_size = 23
    modality_type = "tab_img"

    data_module = KFoldDataModule(
        params,
        modality_type,
        sources,
        batch_size,
    )

    return data_module


def test_data_module_initialization(create_kfold_data_module):
    data_module = create_kfold_data_module

    assert data_module.num_folds == 5
    assert data_module.batch_size == 23
    assert data_module.pred_type == "binary"
    assert data_module.multiclass_dims is None
    assert data_module.subspace_method is None
    assert data_module.layer_mods is None
    assert data_module.max_epochs == 1000


def test_data_preparation(create_kfold_data_module):
    data_module = create_kfold_data_module
    data_module.prepare_data()

    # Add assertions for data preparation, e.g., check if data_dims, dataset, etc. are correctly set.
    assert data_module.data_dims == [2, None, [100, 100]]
    assert data_module.dataset is not None


def test_kfold_split(create_kfold_data_module):
    data_module = create_kfold_data_module
    data_module.prepare_data()
    folds = data_module.kfold_split()

    assert len(folds) == 5  # Check if the correct number of folds is generated


def test_train_dataloader(create_kfold_data_module):
    data_module = create_kfold_data_module
    data_module.prepare_data()
    data_module.setup()

    dataloader = data_module.train_dataloader(
        0
    )  # Assuming you want to test the first fold

    # Add assertions to check if the dataloader returns the expected batch sizes and data.
    assert dataloader.batch_size == 23
    assert type(dataloader.sampler) is torch.utils.data.sampler.RandomSampler


def test_val_dataloader(create_kfold_data_module):
    data_module = create_kfold_data_module
    data_module.prepare_data()
    data_module.setup()

    dataloader = data_module.val_dataloader(
        0
    )  # Assuming you want to test the first fold

    # Add assertions to check if the dataloader returns the expected batch sizes and data.
    assert dataloader.batch_size == 23
    assert type(dataloader.sampler) is torch.utils.data.sampler.SequentialSampler


def test_setup_calls_subspace_method(create_kfold_data_module):
    with patch(
        "fusionlibrary.fusion_models.denoise_tab_img_maps.denoising_autoencoder_subspace_method",
        return_value=MockSubspaceMethod(),
    ) as mock_subspace_method:
        # Initialize the KfoldDataModule
        data_module = create_kfold_data_module
        data_module.subspace_method = mock_subspace_method

        data_module.prepare_data()
        data_module.setup()

        # Assert that the subspace_method class was called 5 times (once for each fold)
        assert mock_subspace_method.call_count == 5


if __name__ == "__main__":
    pytest.main()