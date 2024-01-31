import pytest
from fusilli.data import KFoldGraphDataModule
from .test_TrainTestDataModule import create_test_files, create_test_files_more_features
import torch_geometric
import numpy as np
from unittest.mock import patch, Mock
from pytest_mock import mocker
from sklearn.model_selection import KFold


class MockGraphMakerModule:
    def __init__(self, graph_data):
        self.graph_data = graph_data

    def make_graph(self):
        return self.graph_data


@pytest.fixture
def create_graph_data_module(create_test_files):
    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    image_torch_file_2d = create_test_files["image_torch_file_2d"]

    sources = [tabular1_csv, tabular2_csv, image_torch_file_2d]
    example_fusion_model = Mock()
    example_fusion_model.modality_type = "tabular_tabular"

    data_module = KFoldGraphDataModule(
        num_folds=5,
        fusion_model=example_fusion_model,
        sources=sources,
        graph_creation_method=MockGraphMakerModule,
    )

    return data_module


def test_prepare_data(create_graph_data_module):
    datamodule = create_graph_data_module
    datamodule.prepare_data()
    assert len(datamodule.dataset) > 0
    assert datamodule.data_dims == [2, 2, None]  # Adjust based on your data dimensions
    assert datamodule.layer_mods == None


def test_kfold_split(create_graph_data_module):
    datamodule = create_graph_data_module
    datamodule.prepare_data()
    folds = datamodule.kfold_split()
    assert len(folds) == datamodule.num_folds

    # assert that each fold has one train and one val dataset
    for fold in folds:
        assert len(fold) == 2


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

    datamodule = KFoldGraphDataModule(
        num_folds=5,
        fusion_model=example_fusion_model,
        sources=sources,
        graph_creation_method=MockGraphMakerModule,
        own_kfold_indices=own_folds,
    )

    datamodule.prepare_data()
    folds = datamodule.kfold_split()  # returns list of tuples of datasets

    assert len(folds) == 5  # Check if the correct number of folds is generated

    # check if the correct number of samples is in each fold
    assert len(folds[0][0]) == len(own_folds[0][0])


def test_setup(create_graph_data_module, mocker):
    datamodule = create_graph_data_module
    mocker.patch.object(
        MockGraphMakerModule, "make_graph", return_value="mock_graph_data"
    )
    datamodule.prepare_data()
    datamodule.setup()

    assert len(datamodule.folds) == datamodule.num_folds
    for fold in datamodule.folds:
        assert len(fold) == 3  # graph_data, train_idxs, test_idxs
        assert type(fold[1]) == np.ndarray  # train_idxs
        assert len(fold[1]) > 0
        assert type(fold[2]) == np.ndarray  # test_idxs
        assert len(fold[2]) > 0

        # assert that train_idxs and test_idxs are disjoint
        assert len(set(fold[1]).intersection(set(fold[2]))) == 0

    # check that graph_maker was called exactly num_folds times

    assert MockGraphMakerModule.make_graph.call_count == datamodule.num_folds


def test_get_lightning_module(create_graph_data_module):
    datamodule = create_graph_data_module
    datamodule.prepare_data()
    datamodule.setup()
    lightning_modules = datamodule.get_lightning_module()
    assert len(lightning_modules) == datamodule.num_folds

    for lightning_module in lightning_modules:
        assert lightning_module is not None
        assert lightning_module.data is not None
        assert (
                type(lightning_module) == torch_geometric.data.lightning.LightningNodeData
        )
