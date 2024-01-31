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
            "ID": range(10),  # Different index column name
            "feature1": [1.0] * 10,
            "feature2": [2.0] * 10,
            "prediction_label": [0] * 10,  # Different label column name
        }
    )
    tabular1_data.to_csv(tabular1_csv, index=False)

    tabular2_csv = tmp_dir / "tabular2.csv"
    tabular2_data = pd.DataFrame(
        {
            "ID": range(10),
            "feature3": [3.0] * 10,
            "feature4": [4.0] * 10,
            "prediction_label": [1] * 10,
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


@pytest.fixture(scope="module")
def create_test_files_more_features(tmp_path_factory):
    # Create a temporary directory
    tmp_dir = tmp_path_factory.mktemp("test_data")

    num_people = 36

    # Create sample CSV files with different index and label column names
    tabular1_csv = tmp_dir / "tabular1.csv"
    tabular1_data = pd.DataFrame(
        {
            "ID": range(num_people),  # Different index column name
            "feature1": [1.0] * num_people,
            "feature2": [2.0] * num_people,
            "feature3": [3.0] * num_people,
            "feature4": [4.0] * num_people,
            "feature5": [5.0] * num_people,
            "feature6": [6.0] * num_people,
            "feature7": [7.0] * num_people,
            "feature8": [8.0] * num_people,
            "feature9": [9.0] * num_people,
            "feature10": [10.0] * num_people,
            "prediction_label": [0] * num_people,  # Different label column name
        }
    )
    tabular1_data.to_csv(tabular1_csv, index=False)

    tabular2_csv = tmp_dir / "tabular2.csv"
    tabular2_data = pd.DataFrame(
        {
            "ID": range(num_people),
            "feature1": [1.0] * num_people,
            "feature2": [2.0] * num_people,
            "feature3": [3.0] * num_people,
            "feature4": [4.0] * num_people,
            "feature5": [5.0] * num_people,
            "feature6": [6.0] * num_people,
            "feature7": [7.0] * num_people,
            "feature8": [8.0] * num_people,
            "feature9": [9.0] * num_people,
            "feature10": [10.0] * num_people,
            "feature11": [11.0] * num_people,
            "feature12": [12.0] * num_people,
            "feature13": [13.0] * num_people,
            "feature14": [14.0] * num_people,
            "feature15": [15.0] * num_people,
            "prediction_label": [1] * num_people,
        }
    )
    tabular2_data.to_csv(tabular2_csv, index=False)

    # Create a sample Torch file for image data
    image_data_2d = torch.randn(num_people, 1, 100, 100)
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

    sources = [tabular1_csv, tabular2_csv, image_torch_file_2d]
    batch_size = 8
    test_size = 0.2
    prediction_task = "binary"

    example_fusion_model = Mock()
    example_fusion_model.modality_type = "tabular1"

    # Initialize the TrainTestDataModule
    datamodule = TrainTestDataModule(fusion_model=example_fusion_model,
                                     sources=sources,
                                     output_paths=None,
                                     prediction_task=prediction_task,
                                     batch_size=batch_size,
                                     test_size=test_size,
                                     multiclass_dimensions=None, )

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

    test_size = 0.2
    prediction_task = "binary"
    multiclass_dimensions = None

    sources = [tabular1_csv, tabular2_csv, image_torch_file_2d]
    batch_size = 23

    example_fusion_model = Mock()
    example_fusion_model.modality_type = "tabular_image"

    # Initialize the TrainTestDataModule
    datamodule = TrainTestDataModule(fusion_model=example_fusion_model,
                                     sources=sources,
                                     output_paths=None,
                                     prediction_task=prediction_task,
                                     batch_size=batch_size,
                                     test_size=test_size,
                                     multiclass_dimensions=multiclass_dimensions, )
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
        train_labels = pd.DataFrame([0.3], columns=["prediction_label"])
        return train_latents, train_labels

    def convert_to_latent(self, test_dataset):
        return (
            torch.Tensor([[0.1, 0.2, 0.3]]),
            pd.DataFrame([0.3], columns=["prediction_label"]),
            [3, None, None],
        )


def test_setup_calls_subspace_method(create_test_files):
    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    image_torch_file_2d = create_test_files["image_torch_file_2d"]

    test_size = 0.2
    prediction_task = "binary"
    multiclass_dimensions = None

    sources = [tabular1_csv, tabular2_csv, image_torch_file_2d]
    batch_size = 23

    example_fusion_model = Mock()
    example_fusion_model.modality_type = "tabular_image"

    with patch(
            "fusilli.fusionmodels.tabularimagefusion.denoise_tab_img_maps.denoising_autoencoder_subspace_method",
            return_value=MockSubspaceMethod(),
    ) as mock_subspace_method:
        # Initialize the TrainTestDataModule
        datamodule = TrainTestDataModule(fusion_model=example_fusion_model,
                                         sources=sources,
                                         output_paths=None,
                                         prediction_task=prediction_task,
                                         batch_size=batch_size,
                                         test_size=test_size,
                                         multiclass_dimensions=multiclass_dimensions,
                                         subspace_method=mock_subspace_method)
        datamodule.prepare_data()
        datamodule.setup()

        # Assert that the subspace_method class was called
        mock_subspace_method.assert_called_once_with(
            datamodule=datamodule, max_epochs=datamodule.max_epochs, k=None, train_subspace=True
        )


# Testing that the test indices are correctly input and used instead of a random split
def test_owntestindices(create_test_files_more_features):
    tabular1_csv = create_test_files_more_features["tabular1_csv"]
    tabular2_csv = create_test_files_more_features["tabular2_csv"]
    image_torch_file_2d = create_test_files_more_features["image_torch_file_2d"]

    test_size = 0.2
    prediction_task = "binary"
    multiclass_dimensions = None

    sources = [tabular1_csv, tabular2_csv, image_torch_file_2d]
    batch_size = 23

    example_fusion_model = Mock()
    example_fusion_model.modality_type = "tabular_image"

    # make test indices people 25 to 36
    test_indices = list(range(25, 36))

    datamodule = TrainTestDataModule(fusion_model=example_fusion_model,
                                     sources=sources,
                                     output_paths=None,
                                     prediction_task=prediction_task,
                                     batch_size=batch_size,
                                     test_size=test_size,
                                     multiclass_dimensions=multiclass_dimensions,
                                     test_indices=test_indices)
    datamodule.prepare_data()
    datamodule.setup()

    # check that the test indices are correctly input
    assert datamodule.test_indices == test_indices
    # look at the test dataset
    test_dataset = datamodule.test_dataset
    # check that the test dataset has the correct number of people
    assert len(test_dataset) == len(test_indices)
    # check train dataset
    train_dataset = datamodule.train_dataset
    # check that the train dataset has the correct number of people
    assert len(train_dataset) == 25


# Run pytest
if __name__ == "__main__":
    pytest.main()
