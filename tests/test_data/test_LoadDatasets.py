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

    # optional third tabular modality

    tabular3_csv = tmp_dir / "tabular3.csv"
    tabular3_data = pd.DataFrame(
        {
            "ID": range(10),
            "feature5": [5.0] * 10,
            "feature6": [6.0] * 10,
            "prediction_label": [1] * 10,
        }
    )
    tabular3_data.to_csv(tabular3_csv, index=False)

    bad_tabular1_csv = tmp_dir / "bad_tabular1.csv"
    bad_tabular1_data = pd.DataFrame(
        {
            "ID": range(10),
            "feature3": [3.0] * 10,
            "feature4": [4.0] * 10,
            "label": [1] * 10,
        }
    )
    bad_tabular1_data.to_csv(bad_tabular1_csv, index=False)

    bad_tabular2_csv = tmp_dir / "bad_tabular2.csv"
    bad_tabular2_data = pd.DataFrame(
        {
            "subject": range(10),
            "feature3": [3.0] * 10,
            "feature4": [4.0] * 10,
            "prediction_label": [1] * 10,
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
        "tabular3_csv": tabular3_csv,
        "bad_tabular1_csv": bad_tabular1_csv,
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
    tabular3_csv = create_test_files["tabular3_csv"]
    bad_tabular1_csv = create_test_files["bad_tabular1_csv"]
    bad_tabular2_csv = create_test_files["bad_tabular2_csv"]
    image_torch_file_2d = create_test_files["image_torch_file_2d"]
    image_torch_file_3d = create_test_files["image_torch_file_3d"]

    tab_tab_img2d_data_paths = {
        "tabular1": tabular1_csv,
        "tabular2": tabular2_csv,
        "tabular3": tabular3_csv,
        "image": image_torch_file_2d,
    }

    # Initialize the LoadDatasets class with incorrect column names
    with pytest.raises(
        ValueError, match="The CSV must have an index column named 'ID'."
    ):
        bad_tab_tab_img2d_data_paths = {
            "tabular1": tabular1_csv,
            "tabular2": bad_tabular2_csv,
            "tabular3": tabular3_csv,
            "image": image_torch_file_2d,
        }
        data_loader_2d = LoadDatasets(bad_tab_tab_img2d_data_paths)

    # Initialize the LoadDatasets class with correct column names
    data_loader_2d = LoadDatasets(tab_tab_img2d_data_paths)

    # Load 2D image dataset
    img_dataset_2d, img_dims_2d = data_loader_2d.load_img()
    assert isinstance(img_dataset_2d, CustomDataset)
    # expected a dictionary of dimensions with keys "mod1_dim", "mod2_dim", "mod3_dim", and "img_dim"
    assert isinstance(img_dims_2d, dict)
    assert list(img_dims_2d.keys()) == ["mod1_dim", "mod2_dim", "mod3_dim", "img_dim"]
    assert img_dims_2d["img_dim"] == [32, 32]  # Check data dimensions for 2D image
    # assert the other modalities are None
    assert img_dims_2d["mod1_dim"] == None
    assert img_dims_2d["mod2_dim"] == None
    assert img_dims_2d["mod3_dim"] == None

    # Load 3D image dataset with incorrect label column name
    with pytest.raises(
        ValueError, match="The CSV must have a label column named 'prediction_label'."
    ):
        bad_tab_tab_img3d_data_paths = {
            "tabular1": bad_tabular1_csv,
            "tabular2": tabular2_csv,
            "tabular3": tabular3_csv,
            "image": image_torch_file_3d,
        }
        data_loader_3d = LoadDatasets(bad_tab_tab_img3d_data_paths)

    tab_tab_img3d_data_paths = {
        "tabular1": tabular1_csv,
        "tabular2": tabular2_csv,
        "tabular3": tabular3_csv,
        "image": image_torch_file_3d,
    }
    # Initialize the LoadDatasets class with correct label column name
    data_loader_3d = LoadDatasets(tab_tab_img3d_data_paths)

    # Load 3D image dataset
    img_dataset_3d, img_dims_3d = data_loader_3d.load_img()
    assert isinstance(img_dataset_3d, CustomDataset)
    # expected a dictionary of dimensions with keys "mod1_dim", "mod2_dim", "mod3_dim", and "img_dim"
    assert isinstance(img_dims_3d, dict)
    assert list(img_dims_3d.keys()) == ["mod1_dim", "mod2_dim", "mod3_dim", "img_dim"]
    assert img_dims_3d["img_dim"] == [32, 32, 32]  # Check data dimensions for 3D image
    # assert the other modalities are None
    assert img_dims_3d["mod1_dim"] == None
    assert img_dims_3d["mod2_dim"] == None
    assert img_dims_3d["mod3_dim"] == None

    # tab and 2d image dataset

    tab_img_dataset_2d, tab_img_dims_2d = data_loader_2d.load_tab_and_img()
    assert isinstance(tab_img_dataset_2d, CustomDataset)
    # expected a dictionary of dimensions with keys "mod1_dim", "mod2_dim", "mod3_dim", and "img_dim"
    assert isinstance(tab_img_dims_2d, dict)
    assert list(tab_img_dims_2d.keys()) == [
        "mod1_dim",
        "mod2_dim",
        "mod3_dim",
        "img_dim",
    ]
    assert tab_img_dims_2d["img_dim"] == [32, 32]  # Check data dimensions for 2D image
    assert tab_img_dims_2d["mod1_dim"] == 2
    assert tab_img_dims_2d["mod2_dim"] == None
    assert tab_img_dims_2d["mod3_dim"] == None

    # tab and 3d image dataset

    tab_img_dataset_3d, tab_img_dims_3d = data_loader_3d.load_tab_and_img()
    assert isinstance(tab_img_dataset_3d, CustomDataset)
    # expected a dictionary of dimensions with keys "mod1_dim", "mod2_dim", "mod3_dim", and "img_dim"
    assert isinstance(tab_img_dims_3d, dict)
    assert list(tab_img_dims_3d.keys()) == [
        "mod1_dim",
        "mod2_dim",
        "mod3_dim",
        "img_dim",
    ]
    assert tab_img_dims_3d["img_dim"] == [
        32,
        32,
        32,
    ]  # Check data dimensions for 3D image
    assert tab_img_dims_3d["mod1_dim"] == 2
    assert tab_img_dims_3d["mod2_dim"] == None
    assert tab_img_dims_3d["mod3_dim"] == None

    # tabular-tabular dataset
    tab_tab_dataset, tab_tab_dims = data_loader_2d.load_tabular_tabular()
    assert isinstance(tab_tab_dataset, CustomDataset)
    # expected a dictionary of dimensions with keys "mod1_dim", "mod2_dim", "mod3_dim", and "img_dim"
    assert isinstance(tab_tab_dims, dict)
    assert list(tab_tab_dims.keys()) == ["mod1_dim", "mod2_dim", "mod3_dim", "img_dim"]
    assert tab_tab_dims["img_dim"] == None  # Check data dimensions for 2D image
    assert tab_tab_dims["mod1_dim"] == 2
    assert tab_tab_dims["mod2_dim"] == 2
    assert tab_tab_dims["mod3_dim"] == None

    # tabular-tabular-tabular dataset
    tab_tab_tab_dataset, tab_tab_tab_dims = (
        data_loader_2d.load_tabular_tabular_tabular()
    )
    assert isinstance(tab_tab_tab_dataset, CustomDataset)
    # expected a dictionary of dimensions with keys "mod1_dim", "mod2_dim", "mod3_dim", and "img_dim"
    assert isinstance(tab_tab_tab_dims, dict)
    assert list(tab_tab_tab_dims.keys()) == [
        "mod1_dim",
        "mod2_dim",
        "mod3_dim",
        "img_dim",
    ]
    assert tab_tab_tab_dims["img_dim"] == None  # Check data dimensions for 2D image
    assert tab_tab_tab_dims["mod1_dim"] == 2
    assert tab_tab_tab_dims["mod2_dim"] == 2
    assert tab_tab_tab_dims["mod3_dim"] == 2


# Run pytest
if __name__ == "__main__":
    pytest.main()
