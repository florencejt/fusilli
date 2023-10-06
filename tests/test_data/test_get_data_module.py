import pytest
from fusilli.data import (
    KFoldDataModule,
    CustomDataModule,
    GraphDataModule,
    KFoldGraphDataModule,
    get_data_module,
)
from .test_CustomDataModule import create_test_files, MockSubspaceMethod
from .test_GraphDataModule import MockGraphMakerModule
from torch_geometric.data.lightning import LightningNodeData
import torch
import pandas as pd
import shutil


@pytest.fixture(scope="module")
def create_optional_suffix_files(tmp_path_factory):
    # Create a temporary directory
    tmp_dir = tmp_path_factory.mktemp("test_data")

    # Create sample CSV files with different index and label column names
    tabular1_csv = tmp_dir / "tabular1_optional_suffix.csv"
    tabular1_data = pd.DataFrame(
        {
            "study_id": range(10),  # Different index column name
            "feature1": [1.0] * 10,
            "feature2": [2.0] * 10,
            "pred_label": [0] * 10,  # Different label column name
        }
    )
    tabular1_data.to_csv(tabular1_csv, index=False)

    tabular2_csv = tmp_dir / "tabular2_optional_suffix.csv"
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
    image_torch_file_2d = tmp_dir / "image_data_2d_optional_suffix.pt"
    torch.save(image_data_2d, image_torch_file_2d)

    # Create a sample Torch file for image data
    image_data_3d = torch.randn(10, 1, 100, 100, 100)
    image_torch_file_3d = tmp_dir / "image_data_3d_optional_suffix.pt"
    torch.save(image_data_3d, image_torch_file_3d)

    yield {
        "tabular1_csv": tabular1_csv,
        "tabular2_csv": tabular2_csv,
        "image_torch_file_2d": image_torch_file_2d,
        "image_torch_file_3d": image_torch_file_3d,
    }

    # Clean up temporary files and directories
    shutil.rmtree(tmp_dir)


class MockFusionModel:
    def __init__(self, fusion_type, modality_type, graph_maker):
        self.fusion_type = fusion_type
        self.modality_type = modality_type
        self.graph_maker = graph_maker


def test_get_data_module_custom(create_test_files):
    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    image_torch_file_2d = create_test_files["image_torch_file_2d"]

    fusion_model = MockFusionModel(
        fusion_type="attention",
        modality_type="tab_img",
        graph_maker=None,
    )

    params = {
        "pred_type": "binary",
        "multiclass_dims": None,
        "tabular1_source": tabular1_csv,
        "tabular2_source": tabular2_csv,
        "img_source": image_torch_file_2d,
        "kfold_flag": False,
        "test_size": 0.3,
    }

    # Call the get_data_module function with custom fusion type (non-graph)
    dm = get_data_module(fusion_model, params)

    # Add assertions based on your expectations
    assert isinstance(dm, CustomDataModule)
    assert dm.batch_size == params.get("batch_size", 8)  # default batch size
    assert dm.data_dims == [2, None, [100, 100]]  # Adjust based on your data dimensions
    assert dm.test_size == params.get("test_size", 0.3)


def test_get_k_fold_data_module_custom(create_test_files):
    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    image_torch_file_2d = create_test_files["image_torch_file_2d"]

    fusion_model = MockFusionModel(
        fusion_type="operation",
        modality_type="both_tab",
        graph_maker=None,
    )

    params = {
        "pred_type": "binary",
        "multiclass_dims": None,
        "tabular1_source": tabular1_csv,
        "tabular2_source": tabular2_csv,
        "img_source": image_torch_file_2d,
        "kfold_flag": True,
        "num_k": 7,
    }

    # Call the get_data_module function with custom fusion type (non-graph)
    dm = get_data_module(fusion_model, params, batch_size=16)

    # Add assertions based on your expectations
    assert isinstance(dm, KFoldDataModule)
    assert dm.batch_size == params.get("batch_size", 16)  # changed batch size
    assert dm.data_dims == [2, 2, None]  # Adjust based on your data dimensions


def test_get_graph_data_module(create_test_files):
    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    image_torch_file_2d = create_test_files["image_torch_file_2d"]

    fusion_model = MockFusionModel(
        fusion_type="graph",
        modality_type="both_tab",
        graph_maker=MockGraphMakerModule,
    )

    params = {
        "pred_type": "regression",
        "multiclass_dims": None,
        "tabular1_source": tabular1_csv,
        "tabular2_source": tabular2_csv,
        "img_source": image_torch_file_2d,
        "kfold_flag": False,
        "test_size": 0.3,
    }

    # Call the get_data_module function with custom fusion type (non-graph)
    dm = get_data_module(fusion_model, params)

    # Add assertions based on your expectations
    assert isinstance(dm, LightningNodeData)
    assert dm.data_dims == [2, 2, None]  # Adjust based on your data dimensions


def test_get_kfold_graph_data_module(create_test_files):
    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    image_torch_file_2d = create_test_files["image_torch_file_2d"]

    fusion_model = MockFusionModel(
        fusion_type="graph",
        modality_type="both_tab",
        graph_maker=MockGraphMakerModule,
    )

    params = {
        "pred_type": "regression",
        "multiclass_dims": None,
        "tabular1_source": tabular1_csv,
        "tabular2_source": tabular2_csv,
        "img_source": image_torch_file_2d,
        "kfold_flag": True,
        "num_k": 8,
    }

    # Call the get_data_module function with custom fusion type (non-graph)
    dm = get_data_module(fusion_model, params)

    # Add assertions based on your expectations
    for fold_dm in dm:
        assert isinstance(fold_dm, LightningNodeData)
        assert fold_dm.data_dims == [2, 2, None]  # Adjust based on your data dimensions


# test get_data_module works with the optional suffix and accesses the correct csvs
def test_optional_suffix(create_optional_suffix_files):
    optional_suffix = "_optional_suffix"

    tabular1_csv = create_optional_suffix_files["tabular1_csv"]
    tabular2_csv = create_optional_suffix_files["tabular2_csv"]
    image_torch_file_2d = create_optional_suffix_files["image_torch_file_2d"]

    fusion_model = MockFusionModel(
        fusion_type="attention",
        modality_type="tab_img",
        graph_maker=None,
    )

    params = {
        "kfold_flag": False,
        "test_size": 0.3,
        "pred_type": "binary",
        "multiclass_dims": None,
    }
    params["tabular1_source" + optional_suffix] = tabular1_csv
    params["tabular2_source" + optional_suffix] = tabular2_csv
    params["img_source" + optional_suffix] = image_torch_file_2d

    # Call the get_data_module function with custom fusion type (non-graph)
    dm = get_data_module(fusion_model, params, optional_suffix=optional_suffix)

    # with incorrect optional suffix specified
    with pytest.raises(KeyError):
        dm = get_data_module(fusion_model, params, optional_suffix="_wrong_suffix")

    # Add assertions based on your expectations
    assert isinstance(dm, CustomDataModule)
    assert dm.batch_size == params.get("batch_size", 8)  # default batch size
    assert dm.data_dims == [2, None, [100, 100]]  # Adjust based on your data dimensions
    assert dm.test_size == params.get("test_size", 0.3)
