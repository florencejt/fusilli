import pytest
import torch.nn as nn
from fusilli.utils import model_modifier
from fusilli.fusionmodels.tabularfusion.edge_corr_gnn import (
    EdgeCorrGNN,
    EdgeCorrGraphMaker,
)
from fusilli.fusionmodels.tabularfusion.attention_weighted_GNN import (
    AttentionWeightedGNN,
    AttentionWeightedGraphMaker
)
from tests.test_data.test_TrainTestDataModule import create_test_files
from fusilli.data import TrainTestGraphDataModule
from torch_geometric.nn import GCNConv, ChebConv
from unittest.mock import patch, Mock
from lightning.pytorch.callbacks import EarlyStopping


@pytest.fixture
def model_instance_EdgeCorrGNN():
    prediction_task = "regression"
    data_dims = [10, 15, None]
    multiclass_dimensions = None
    return EdgeCorrGNN(prediction_task, data_dims, multiclass_dimensions)


@pytest.fixture
def model_instance_EdgeCorrGraphMaker(create_test_files):
    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    image_torch_file_2d = create_test_files["image_torch_file_2d"]

    sources = [tabular1_csv, tabular2_csv, image_torch_file_2d]
    example_fusion_model = Mock()
    example_fusion_model.modality_type = "tabular_tabular"

    # Initialize the TrainTestDataModule
    dm = TrainTestGraphDataModule(
        example_fusion_model,
        sources,
        EdgeCorrGraphMaker,
        test_size=0.3
    )
    dm.prepare_data()

    return EdgeCorrGraphMaker(dm.dataset)


@pytest.fixture
def model_instance_AttentionWeightedGNN():
    prediction_task = "regression"
    data_dims = [10, 15, None]
    multiclass_dimensions = None
    return AttentionWeightedGNN(prediction_task, data_dims, multiclass_dimensions)


@pytest.fixture
def model_instance_AttentionWeightedGraphMaker(create_test_files):
    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    image_torch_file_2d = create_test_files["image_torch_file_2d"]

    params = {
        "test_size": 0.2,
    }

    sources = [tabular1_csv, tabular2_csv, image_torch_file_2d]
    example_fusion_model = Mock()
    example_fusion_model.modality_type = "tabular_tabular"

    # Initialize the TrainTestDataModule
    dm = TrainTestGraphDataModule(
        example_fusion_model,
        sources,
        AttentionWeightedGraphMaker,
        test_size=0.3
    )
    dm.prepare_data()

    return AttentionWeightedGraphMaker(dm.dataset)


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
    "AttentionWeightedGNN": {
        "graph_conv_layers": nn.Sequential(
            ChebConv(23, 50, K=3),
            ChebConv(50, 100, K=3),
            ChebConv(100, 130, K=3),
        ),
        "dropout_prob": 0.4,
    },
    "AttentionWeightedGraphMaker": {"early_stop_callback":
                                        EarlyStopping(monitor="val_loss", ),
                                    "edge_probability_threshold": 80,
                                    "attention_MLP_test_size": 0.3,
                                    "AttentionWeightingMLPInstance.weighting_layers": nn.ModuleDict(
                                        {
                                            "Layer 1": nn.Sequential(nn.Linear(4, 100),
                                                                     nn.ReLU()),
                                            "Layer 2": nn.Sequential(nn.Linear(100, 75),
                                                                     nn.ReLU()),
                                            "Layer 3": nn.Sequential(nn.Linear(75, 75),
                                                                     nn.ReLU()),
                                            "Layer 4": nn.Sequential(nn.Linear(75, 100),
                                                                     nn.ReLU()),
                                            "Layer 5": nn.Sequential(nn.Linear(100, 4),
                                                                     nn.ReLU()),
                                        }
                                    )},
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
    "AttentionWeightedGNN": {
        "graph_conv_layers": nn.ModuleDict(
            {"conv1": ChebConv(23, 50, K=3),
             "conv2": ChebConv(50, 100, K=3),
             "conv3": ChebConv(100, 130, K=3), }
        ),
        "dropout_prob": 1,
    },
    "AttentionWeightedGraphMaker": {"early_stop_callback":
                                        "earlystopping",
                                    "edge_probability_threshold": 80.3,
                                    "attention_MLP_test_size": 1,
                                    "AttentionWeightingMLPInstance.weighting_layers": nn.Sequential(
                                        nn.Linear(4, 100),
                                        nn.ReLU(), nn.Linear(100, 75),
                                        nn.ReLU(), nn.Linear(75, 75),
                                        nn.ReLU(), nn.Linear(75, 100),
                                        nn.ReLU(), nn.Linear(100, 4),
                                        nn.ReLU())
                                    },
}

incorrect_data_ranges_modifications = {
    "EdgeCorrGNN": {
        "dropout_prob": 1.5,
    },
    "EdgeCorrGraphMaker": {"threshold": -0.4},
    "AttentionWeightedGNN": {
        "dropout_prob": 1.5,
    },
    "AttentionWeightedGraphMaker": {
        "edge_probability_threshold": 120,
        "attention_MLP_test_size": 1.5, },
}

model_instances = [
    ("EdgeCorrGNN", "model_instance_EdgeCorrGNN"),
    ("EdgeCorrGraphMaker", "model_instance_EdgeCorrGraphMaker"),
    ("AttentionWeightedGNN", "model_instance_AttentionWeightedGNN"),
    ("AttentionWeightedGraphMaker", "model_instance_AttentionWeightedGraphMaker"),
]


# test to see if its doing it correctly
@pytest.mark.parametrize("model_name, model_fixture", model_instances)
def test_correct_modification(model_name, model_fixture, request):
    # "modification" may have been modified (ironically) by the calc_fused_layers method
    # in some of the models. This is to ensure that the input to the layers is consistent
    # with either the input data dimensions or the output dimensions of the previous layer.

    # This test is to ensure that the modification has been applied at all, not to
    # check the modification itself is exactly what it was in the dictionary
    model_fixture = request.getfixturevalue(model_fixture)

    original_model = model_fixture

    modified_model = model_modifier.modify_model_architecture(
        model_fixture, correct_modifications
    )

    # check that the model has been modified
    assert modified_model is not None

    for key, modification in correct_modifications.get(model_name, {}).items():
        attr_name = key.split(".")[-1]
        nested_attr = model_modifier.get_nested_attr(modified_model, key)

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
            model_modifier.modify_model_architecture(
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
            model_modifier.modify_model_architecture(
                request.getfixturevalue(model_fixture),
                individual_modification,
            )
