import pytest
import torch
from fusilli.data import KFoldDataModule
from .test_TrainTestDataModule import create_test_files, MockSubspaceMethod, create_test_files_more_features
from unittest.mock import patch, Mock
from sklearn.model_selection import KFold


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

    example_fusion_model = Mock()
    example_fusion_model.modality_type = "tabular_image"

    data_module = KFoldDataModule(
        fusion_model=example_fusion_model,
        sources=sources,
        output_paths={},
        prediction_task=params["pred_type"],
        multiclass_dimensions=params["multiclass_dims"],
        num_folds=params["num_k"],
        batch_size=batch_size,
    )

    return data_module


def test_data_module_initialization(create_kfold_data_module):
    data_module = create_kfold_data_module

    assert data_module.num_folds == 5
    assert data_module.batch_size == 23
    assert data_module.prediction_task == "binary"
    assert data_module.multiclass_dimensions is None
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


def test_kfold_split_own_indices(create_test_files_more_features):
    tabular1_csv = create_test_files_more_features["tabular1_csv"]
    tabular2_csv = create_test_files_more_features["tabular2_csv"]
    image_torch_file_2d = create_test_files_more_features["image_torch_file_2d"]

    prediction_task = "binary"
    multiclass_dimensions = None

    sources = [tabular1_csv, tabular2_csv, image_torch_file_2d]

    # specifying own kfold indices using a non random split
    own_folds = [(train_index, test_index) for train_index, test_index in KFold(n_splits=5).split(range(36))]

    example_fusion_model = Mock()
    example_fusion_model.modality_type = "tabular_image"

    datamodule = KFoldDataModule(
        fusion_model=example_fusion_model,
        sources=sources,
        output_paths={},
        prediction_task=prediction_task,
        multiclass_dimensions=multiclass_dimensions,
        num_folds=5,
        own_kfold_indices=own_folds,
        batch_size=9,
    )

    datamodule.prepare_data()
    folds = datamodule.kfold_split()  # returns list of tuples of datasets

    assert len(folds) == 5  # Check if the correct number of folds is generated

    # check if the correct number of samples is in each fold
    assert len(folds[0][0]) == len(own_folds[0][0])


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
            "fusilli.fusionmodels.tabularimagefusion.denoise_tab_img_maps.denoising_autoencoder_subspace_method",
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
