import pytest
from fusilli.data import TrainTestGraphDataModule
from .test_TrainTestDataModule import create_test_files, create_test_files_more_features
from pytest import approx
from pytest_mock import mocker


class MockGraphMakerModule:
    def __init__(self, graph_data):
        self.graph_data = graph_data

    def make_graph(self):
        return self.graph_data


@pytest.fixture
def create_graph_data_module(create_test_files):
    params = {
        "test_size": 0.3,
        "pred_type": "binary",
        "multiclass_dims": None,
    }

    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    image_torch_file_2d = create_test_files["image_torch_file_2d"]

    sources = [tabular1_csv, tabular2_csv, image_torch_file_2d]
    batch_size = 23

    class example_fusion_model:
        modality_type = "tabular_tabular"

        def __init__(self):
            pass

    data_module = TrainTestGraphDataModule(
        fusion_model=example_fusion_model,
        sources=sources,
        graph_creation_method=MockGraphMakerModule,
        test_size=params["test_size"],
    )

    return data_module


# Write test functions
def test_prepare_data(create_graph_data_module):
    # Create a GraphDataModule instance with test parameters
    datamodule = create_graph_data_module

    # Call the prepare_data method
    datamodule.prepare_data()

    # Add assertions to check the expected behavior
    assert len(datamodule.dataset) > 0
    assert datamodule.data_dims == [2, 2, None]  # Adjust based on your data dimensions


def test_setup(create_graph_data_module, mocker):
    # Create a GraphDataModule instance with test parameters
    datamodule = create_graph_data_module
    datamodule.prepare_data()

    mocker.patch.object(
        MockGraphMakerModule, "make_graph", return_value="mock_graph_data"
    )

    # Call the setup method
    datamodule.setup()

    # Assert length of train and test indices is consistent with the test size
    assert len(datamodule.train_idxs) == approx(
        (1 - datamodule.test_size) * len(datamodule.dataset)
    )
    assert len(datamodule.test_idxs) > 0

    # Assert that the graph data is not None and the graph maker is called
    MockGraphMakerModule.make_graph.assert_called_once()
    assert datamodule.graph_data is not None

    # Check if the train and test indices are disjoint
    assert set(datamodule.train_idxs).intersection(set(datamodule.test_idxs)) == set()


def test_get_lightning_module(create_graph_data_module):
    # Create a GraphDataModule instance with test parameters
    datamodule = create_graph_data_module
    datamodule.prepare_data()
    datamodule.setup()

    # Call the get_lightning_module method
    lightning_module = datamodule.get_lightning_module()

    # Add assertions to check the expected behavior of the lightning module
    assert lightning_module is not None


# Testing the TrainTestGraphDataModule class for the case where the user specifies their own test indices
def test_owntestindices(create_test_files_more_features):
    params = {
        "test_size": 0.3,
        "pred_type": "binary",
        "multiclass_dims": None,
    }

    tabular1_csv = create_test_files_more_features["tabular1_csv"]
    tabular2_csv = create_test_files_more_features["tabular2_csv"]
    image_torch_file_2d = create_test_files_more_features["image_torch_file_2d"]

    sources = [tabular1_csv, tabular2_csv, image_torch_file_2d]
    batch_size = 23

    # make test indices people 25 to 36
    test_indices = list(range(25, 36))

    class example_fusion_model:
        modality_type = "tabular_tabular"

        def __init__(self):
            pass

    data_module = TrainTestGraphDataModule(
        fusion_model=example_fusion_model,
        sources=sources,
        graph_creation_method=MockGraphMakerModule,
        test_size=params["test_size"],
        own_test_indices=test_indices,
    )

    data_module.prepare_data()
    data_module.setup()
    lightning_module = data_module.get_lightning_module()

    # check that the test indices are the same as the ones we specified
    assert data_module.test_idxs == test_indices


if __name__ == "__main__":
    pytest.main()
