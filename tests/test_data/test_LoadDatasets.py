import pytest
import pandas as pd
import torch
from fusilli.data import LoadDatasets, CustomDataset  # Import your classes
import shutil


@pytest.fixture(scope="module")
def create_test_files(tmp_path_factory):
    # Create a temporary directory
    tmp_dir = tmp_path_factory.mktemp("test_data")

    # Create sample CSV files with different index and label column names
    tabular1_csv = tmp_dir / "tabular1.csv"
    tabular1_data = pd.DataFrame(
        {
            "patient_id": range(10),  # Different index column name
            "feature1": [1.0] * 10,
            "feature2": [2.0] * 10,
            "label": [0] * 10,  # Different label column name
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

    bad_tabular2_csv = tmp_dir / "bad_tabular2.csv"
    bad_tabular2_data = pd.DataFrame(
        {
            "ID": range(10),
            "feature3": [3.0] * 10,
            "feature4": [4.0] * 10,
            "label_feature": [1] * 10,
        }
    )
    bad_tabular2_data.to_csv(bad_tabular2_csv, index=False)

    # Create a sample Torch file for image data
    image_data_2d = torch.randn(10, 3, 32, 32)
    image_torch_file_2d = tmp_dir / "image_data_2d.pt"
    torch.save(image_data_2d, image_torch_file_2d)

    image_data_3d = torch.randn(10, 1, 32, 32, 32)
    image_torch_file_3d = tmp_dir / "image_data_3d.pt"
    torch.save(image_data_3d, image_torch_file_3d)

    yield {
        "tabular1_csv": tabular1_csv,
        "tabular2_csv": tabular2_csv,
        "bad_tabular2_csv": bad_tabular2_csv,
        "image_torch_file_2d": image_torch_file_2d,
        "image_torch_file_3d": image_torch_file_3d,
    }

    # Clean up temporary files and directories
    shutil.rmtree(tmp_dir)


# Test the LoadDatasets class with incorrect column names
def test_load_datasets_with_incorrect_column_names(create_test_files):
    # Get file paths from the fixture
    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    bad_tabular2_csv = create_test_files["bad_tabular2_csv"]
    image_torch_file_2d = create_test_files["image_torch_file_2d"]
    image_torch_file_3d = create_test_files["image_torch_file_3d"]

    # Initialize the LoadDatasets class with incorrect column names
    with pytest.raises(
            ValueError, match="The CSV must have an index column named 'ID'."
    ):
        data_loader_2d = LoadDatasets([tabular1_csv, tabular2_csv, image_torch_file_2d])

    # Initialize the LoadDatasets class with correct column names
    data_loader_2d = LoadDatasets([tabular2_csv, tabular2_csv, image_torch_file_2d])

    # Load 2D image dataset
    img_dataset_2d, img_dims_2d = data_loader_2d.load_img()
    assert isinstance(img_dataset_2d, CustomDataset)
    assert img_dims_2d == [None, None, [32, 32]]  # Check data dimensions for 2D image

    # Load 3D image dataset with incorrect label column name
    with pytest.raises(
            ValueError, match="The CSV must have a label column named 'prediction_label'."
    ):
        data_loader_3d = LoadDatasets(
            [bad_tabular2_csv, tabular2_csv, image_torch_file_3d]
        )

    # Initialize the LoadDatasets class with correct label column name
    data_loader_3d = LoadDatasets([tabular2_csv, tabular2_csv, image_torch_file_3d])

    # Load 3D image dataset
    img_dataset_3d, img_dims_3d = data_loader_3d.load_img()
    assert isinstance(img_dataset_3d, CustomDataset)
    assert img_dims_3d == [
        None,
        None,
        [32, 32, 32],
    ]  # Check data dimensions for 3D image

    tab_img_dataset_2d, tab_img_dims_2d = data_loader_2d.load_tab_and_img()
    assert isinstance(tab_img_dataset_2d, CustomDataset)
    assert tab_img_dims_2d == [2, None, [32, 32]]

    tab_img_dataset_3d, tab_img_dims_3d = data_loader_3d.load_tab_and_img()
    assert isinstance(tab_img_dataset_3d, CustomDataset)
    assert tab_img_dims_3d == [2, None, [32, 32, 32]]


# Run pytest
if __name__ == "__main__":
    pytest.main()
