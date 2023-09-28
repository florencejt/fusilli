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
