import pytest
import torch.nn as nn
from fusionlibrary.train_functions import modify_model_architecture, get_nested_attr
from fusionlibrary.fusion_models.edge_corr_gnn import EdgeCorrGNN, EdgeCorrGraphMaker
from tests.test_datamodules.test_CustomDataModule import create_test_files
from fusionlibrary.datamodules import GraphDataModule
from torch_geometric.nn import GCNConv


@pytest.fixture
def model_instance_EdgeCorrGNN():
    pred_type = "regression"
    data_dims = [10, 15, None]
    params = {}
    return EdgeCorrGNN(pred_type, data_dims, params)


@pytest.fixture
def model_instance_EdgeCorrGraphMaker(create_test_files):
    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    image_torch_file_2d = create_test_files["image_torch_file_2d"]

    params = {
        "test_size": 0.2,
    }

    sources = [tabular1_csv, tabular2_csv, image_torch_file_2d]
    modality_type = "both_tab"

    # Initialize the CustomDataModule
    dm = GraphDataModule(params, modality_type, sources, EdgeCorrGraphMaker)
    dm.prepare_data()

    return EdgeCorrGraphMaker(dm.dataset)


correct_modifications = {
    "EdgeCorrGNN": {
        "graph_conv_layers": nn.Sequential(
            GCNConv(23, 50),
            GCNConv(50, 100),
            GCNConv(100, 130),
        ),
        "dropout_prob": 0.4,
    },
    "EdgeCorrGraphMaker": {"threshold": 0.6},
}

incorrect_data_type_modifications = {
    "EdgeCorrGNN": {
        "graph_conv_layers": nn.ModuleDict(
            {
                "conv1": GCNConv(23, 50),
                "conv2": GCNConv(50, 100),
                "conv3": GCNConv(100, 130),
            }
        ),
        "dropout_prob": 1,
    },
    "EdgeCorrGraphMaker": {"threshold": 0},
}

incorrect_data_ranges_modifications = {
    "EdgeCorrGNN": {
        "dropout_prob": 1.5,
    },
    "EdgeCorrGraphMaker": {"threshold": -0.4},
}

model_instances = [
    ("EdgeCorrGNN", "model_instance_EdgeCorrGNN"),
    ("EdgeCorrGraphMaker", "model_instance_EdgeCorrGraphMaker"),
]


# test to see if its doing it correctly
@pytest.mark.parametrize("model_name, model_fixture", model_instances)
def test_correct_modification(model_name, model_fixture, request):
    model_fixture = request.getfixturevalue(model_fixture)

    original_model = model_fixture

    modified_model = modify_model_architecture(model_fixture, correct_modifications)

    # check that the model has been modified
    assert modified_model is not None

    for key, modification in correct_modifications.get(model_name, {}).items():
        attr_name = key.split(".")[-1]
        nested_attr = get_nested_attr(modified_model, key)

        if hasattr(nested_attr, attr_name):
            assert getattr(nested_attr, attr_name) == modification
        else:
            assert nested_attr == modification

    if hasattr(original_model, "final_prediction"):
        assert (
            modified_model.final_prediction[-1].out_features
            == original_model.final_prediction[-1].out_features
        )


# test to see if it throws an error for incorrect data types
@pytest.mark.parametrize("model_name, model_fixture", model_instances)
def test_incorrect_data_types_modification(model_name, model_fixture, request):
    for key, modification in incorrect_data_type_modifications.get(
        model_name, {}
    ).items():
        individual_modification = {model_name: {key: modification}}

        with pytest.raises(
            TypeError, match="Incorrect data type for the modifications"
        ):
            modify_model_architecture(
                request.getfixturevalue(model_fixture),
                individual_modification,
            )


# test to see if it throws an error for incorrect data ranges
@pytest.mark.parametrize("model_name, model_fixture", model_instances)
def test_incorrect_data_ranges_modification(model_name, model_fixture, request):
    for key, modification in incorrect_data_ranges_modifications.get(
        model_name, {}
    ).items():
        individual_modification = {model_name: {key: modification}}

        with pytest.raises(ValueError, match="Incorrect attribute range:"):
            modify_model_architecture(
                request.getfixturevalue(model_fixture),
                individual_modification,
            )
