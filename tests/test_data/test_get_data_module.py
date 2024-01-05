import pytest
from fusilli.data import (
    KFoldDataModule,
    TrainTestDataModule,
    TrainTestGraphDataModule,
    KFoldGraphDataModule,
    prepare_fusion_data,
)
from .test_TrainTestDataModule import create_test_files, MockSubspaceMethod
from .test_TrainTestGraphDataModule import MockGraphMakerModule
from torch_geometric.data.lightning import LightningNodeData
import torch
import pandas as pd
import shutil


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
        modality_type="tabular_image",
        graph_maker=None,
    )

    data_paths = {"tabular1": tabular1_csv,
                  "tabular2": tabular2_csv,
                  "image": image_torch_file_2d,
                  }

    # Call the prepare_fusion_data function with custom fusion type (non-graph)
    dm = prepare_fusion_data(prediction_task="binary", fusion_model=fusion_model, data_paths=data_paths,
                             output_paths=None, test_size=0.3, batch_size=8, multiclass_dims=None, )

    # Add assertions based on your expectations
    assert isinstance(dm, TrainTestDataModule)
    assert dm.batch_size == 8  # default batch size
    assert dm.data_dims == [2, None, [100, 100]]  # Adjust based on your data dimensions
    assert dm.test_size == 0.3


def test_get_k_fold_data_module_custom(create_test_files):
    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    image_torch_file_2d = create_test_files["image_torch_file_2d"]

    fusion_model = MockFusionModel(
        fusion_type="operation",
        modality_type="tabular_tabular",
        graph_maker=None,
    )

    data_paths = {"tabular1": tabular1_csv,
                  "tabular2": tabular2_csv,
                  "image": image_torch_file_2d,
                  }

    # Call the prepare_fusion_data function with custom fusion type (non-graph)
    dm = prepare_fusion_data(
        prediction_task="binary",
        fusion_model=fusion_model,
        data_paths=data_paths,
        output_paths=None,
        kfold=True,
        num_folds=7,
        batch_size=16)

    # Add assertions based on your expectations
    assert isinstance(dm, KFoldDataModule)
    assert dm.batch_size == 16  # changed batch size
    assert dm.data_dims == [2, 2, None]  # Adjust based on your data dimensions


def test_get_graph_data_module(create_test_files):
    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    image_torch_file_2d = create_test_files["image_torch_file_2d"]

    fusion_model = MockFusionModel(
        fusion_type="graph",
        modality_type="tabular_tabular",
        graph_maker=MockGraphMakerModule,
    )

    data_paths = {"tabular1": tabular1_csv,
                  "tabular2": tabular2_csv,
                  "image": image_torch_file_2d,
                  }

    # Call the prepare_fusion_data function with custom fusion type (non-graph)
    dm = prepare_fusion_data(prediction_task="regression",
                             fusion_model=fusion_model,
                             data_paths=data_paths,
                             output_paths=None,
                             multiclass_dims=None,
                             test_size=0.3,
                             )

    # Add assertions based on your expectations
    assert isinstance(dm, LightningNodeData)
    assert dm.data_dims == [2, 2, None]  # Adjust based on your data dimensions


def test_get_kfold_graph_data_module(create_test_files):
    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    image_torch_file_2d = create_test_files["image_torch_file_2d"]

    fusion_model = MockFusionModel(
        fusion_type="graph",
        modality_type="tabular_tabular",
        graph_maker=MockGraphMakerModule,
    )

    data_paths = {"tabular1": tabular1_csv,
                  "tabular2": tabular2_csv,
                  "image": image_torch_file_2d,
                  }

    # Call the prepare_fusion_data function with custom fusion type (non-graph)
    dm = prepare_fusion_data(prediction_task="regression",
                             fusion_model=fusion_model,
                             data_paths=data_paths,
                             output_paths=None,
                             kfold=True,
                             num_folds=8,
                             batch_size=16)

    # Add assertions based on your expectations
    for fold_dm in dm:
        assert isinstance(fold_dm, LightningNodeData)
        assert fold_dm.data_dims == [2, 2, None]  # Adjust based on your data dimensions
