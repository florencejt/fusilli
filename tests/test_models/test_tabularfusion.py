"""
Tests for the models in fusilli.fusionmodels.tabularfusion.*
"""

import pytest
import torch
import torch.nn as nn

from fusilli.utils.model_chooser import import_chosen_fusion_models
from fusilli.fusionmodels.base_model import ParentFusionModel

fusion_models = import_chosen_fusion_models({
    "modality_type": ["tabular_tabular"]
}, skip_models=["MCVAE_tab"])

fusion_model_names = [model.__name__ for model in fusion_models]
# make into dict
fusion_model_dict = {fusion_model_names[i]: fusion_models[i] for i in range(len(fusion_model_names))}


def test_ConcatTabularFeatureMaps():
    test_model = fusion_model_dict["ConcatTabularFeatureMaps"]

    # attributes available pre-initialisation
    assert hasattr(test_model, "method_name")
    assert test_model.method_name == "Concatenating tabular feature maps"
    assert hasattr(test_model, "modality_type")
    assert test_model.modality_type == "tabular_tabular"
    assert hasattr(test_model, "fusion_type")
    assert test_model.fusion_type == "operation"

    test_model = test_model(prediction_task="binary", data_dims=[10, 14, None], multiclass_dimensions=None)

    # initialising
    assert isinstance(test_model, nn.Module)
    assert isinstance(test_model, ParentFusionModel)
    assert hasattr(test_model, "prediction_task")
    assert test_model.prediction_task == "binary"
    assert hasattr(test_model, "mod1_layers")
    assert test_model.mod1_layers['layer 1'][0].in_features == 10
    assert hasattr(test_model, "mod2_layers")
    assert test_model.mod2_layers['layer 1'][0].in_features == 14
    assert hasattr(test_model, "fused_dim")
    assert test_model.fused_dim == test_model.mod1_layers['layer 5'][0].out_features + test_model.mod2_layers[
        'layer 5'][0].out_features
    assert hasattr(test_model, "fused_layers")
    assert hasattr(test_model, "final_prediction")
    assert hasattr(test_model, "forward")

    # forward pass
    test_input = (torch.randn(8, 10), torch.randn(8, 14))
    test_output = test_model.forward(test_input)
    assert isinstance(test_output, list)
    assert test_output[0].shape == torch.Size([8, 1])
    assert len(test_output) == 1

    # wrong input
    # - too many dimensions
    with pytest.raises(ValueError, match=r"Wrong number of inputs"):
        test_model.forward((torch.randn(8, 10), torch.randn(8, 14), torch.randn(8, 10)))
    # - too few dimensions
    with pytest.raises(ValueError, match=r"Wrong number of inputs"):
        test_model.forward(())
    with pytest.raises(ValueError, match=r"Wrong number of inputs"):
        test_model.forward(tuple(torch.randn(8, 10)))
    with pytest.raises(TypeError, match=r"Wrong input type for model! Expected tuple"):
        test_model.forward(torch.randn(8, 10))


# fusilli.fusionmodels.tabularfusion.concat_data.ConcatTabularData
def test_ConcatTabularData():
    test_model = fusion_model_dict["ConcatTabularData"]

    # attributes available pre-initialisation
    assert hasattr(test_model, "method_name")
    assert test_model.method_name == "Concatenating tabular data"
    assert hasattr(test_model, "modality_type")
    assert test_model.modality_type == "tabular_tabular"
    assert hasattr(test_model, "fusion_type")
    assert test_model.fusion_type == "operation"

    test_model = test_model(prediction_task="binary", data_dims=[10, 14, None], multiclass_dimensions=None)

    # initialising
    assert isinstance(test_model, nn.Module)
    assert isinstance(test_model, ParentFusionModel)
    assert hasattr(test_model, "prediction_task")
    assert test_model.prediction_task == "binary"
    assert hasattr(test_model, "mod1_layers") == False
    assert hasattr(test_model, "mod2_layers") == False
    assert hasattr(test_model, "fused_dim")
    assert test_model.fused_dim == 10 + 14
    assert hasattr(test_model, "fused_layers")
    assert test_model.fused_layers[0].in_features == 10 + 14
    assert hasattr(test_model, "final_prediction")
    assert hasattr(test_model, "forward")

    # forward pass
    test_input = (torch.randn(8, 10), torch.randn(8, 14))
    test_output = test_model.forward(test_input)
    assert isinstance(test_output, list)
    assert test_output[0].shape == torch.Size([8, 1])
    assert len(test_output) == 1

    # wrong input
    # - too many dimensions
    with pytest.raises(ValueError, match=r"Wrong number of inputs"):
        test_model.forward((torch.randn(8, 10), torch.randn(8, 14), torch.randn(8, 10)))
    # - too few dimensions
    with pytest.raises(ValueError, match=r"Wrong number of inputs"):
        test_model.forward(())
    with pytest.raises(ValueError, match=r"Wrong number of inputs"):
        test_model.forward(tuple(torch.randn(8, 10)))
    with pytest.raises(TypeError, match=r"Wrong input type for model! Expected tuple"):
        test_model.forward(torch.randn(8, 10))


# fusilli.fusionmodels.tabularfusion.channelwise_att.TabularChannelWiseMultiAttention

def test_TabularChannelWiseMultiAttention():
    test_model = fusion_model_dict["TabularChannelWiseMultiAttention"]

    # attributes available pre-initialisation
    assert hasattr(test_model, "method_name")
    assert test_model.method_name == "Channel-wise multiplication net (tabular)"
    assert hasattr(test_model, "modality_type")
    assert test_model.modality_type == "tabular_tabular"
    assert hasattr(test_model, "fusion_type")
    assert test_model.fusion_type == "attention"

    test_model = test_model(prediction_task="binary", data_dims=[10, 14, None], multiclass_dimensions=None)

    # initialising
    assert isinstance(test_model, nn.Module)
    assert isinstance(test_model, ParentFusionModel)
    assert hasattr(test_model, "prediction_task")
    assert test_model.prediction_task == "binary"
    assert hasattr(test_model, "mod1_layers")
    assert test_model.mod1_layers['layer 1'][0].in_features == 10
    assert hasattr(test_model, "mod2_layers")
    assert test_model.mod2_layers['layer 1'][0].in_features == 14
    assert hasattr(test_model, "fused_dim")
    assert test_model.fused_dim == test_model.mod2_layers['layer 5'][0].out_features
    assert hasattr(test_model, "fused_layers")
    assert test_model.fused_layers[0].in_features == test_model.mod2_layers['layer 5'][0].out_features
    assert hasattr(test_model, "final_prediction")
    assert hasattr(test_model, "forward")

    # forward pass
    test_input = (torch.randn(8, 10), torch.randn(8, 14))
    test_output = test_model.forward(test_input)
    assert isinstance(test_output, list)
    assert test_output[0].shape == torch.Size([8, 1])
    assert len(test_output) == 1

    # wrong input
    # - too many dimensions
    with pytest.raises(ValueError, match=r"Wrong number of inputs"):
        test_model.forward((torch.randn(8, 10), torch.randn(8, 14), torch.randn(8, 10)))
    # - too few dimensions
    with pytest.raises(ValueError, match=r"Wrong number of inputs"):
        test_model.forward(())
    with pytest.raises(ValueError, match=r"Wrong number of inputs"):
        test_model.forward(tuple(torch.randn(8, 10)))
    with pytest.raises(TypeError, match=r"Wrong input type for model! Expected tuple"):
        test_model.forward(torch.randn(8, 10))


# fusilli.fusionmodels.tabularfusion.crossmodal_att.TabularCrossmodalMultiheadAttention
def test_TabularCrossmodalMultiheadAttention():
    test_model = fusion_model_dict["TabularCrossmodalMultiheadAttention"]

    # attributes available pre-initialisation
    assert hasattr(test_model, "method_name")
    assert test_model.method_name == "Tabular Crossmodal multi-head attention"
    assert hasattr(test_model, "modality_type")
    assert test_model.modality_type == "tabular_tabular"
    assert hasattr(test_model, "fusion_type")
    assert test_model.fusion_type == "attention"

    test_model = test_model(prediction_task="binary", data_dims=[10, 14, None], multiclass_dimensions=None)

    # initialising
    assert isinstance(test_model, nn.Module)
    assert isinstance(test_model, ParentFusionModel)
    assert hasattr(test_model, "prediction_task")
    assert test_model.prediction_task == "binary"
    assert hasattr(test_model, "mod1_layers")
    assert test_model.mod1_layers['layer 1'][0].in_features == 10
    assert hasattr(test_model, "mod2_layers")
    assert test_model.mod2_layers['layer 1'][0].in_features == 14
    assert hasattr(test_model, "fused_dim")
    assert test_model.fused_dim == test_model.mod1_layers['layer 5'][0].out_features
    assert hasattr(test_model, "final_prediction")
    assert hasattr(test_model, "forward")
    assert hasattr(test_model, "attention")

    # forward pass
    test_input = (torch.randn(8, 10), torch.randn(8, 14))
    test_output = test_model.forward(test_input)
    assert isinstance(test_output, list)
    assert test_output[0].shape == torch.Size([8, 1])
    assert len(test_output) == 1

    # wrong input
    # - too many dimensions
    with pytest.raises(ValueError, match=r"Wrong number of inputs"):
        test_model.forward((torch.randn(8, 10), torch.randn(8, 14), torch.randn(8, 10)))
    # - too few dimensions
    with pytest.raises(ValueError, match=r"Wrong number of inputs"):
        test_model.forward(())
    with pytest.raises(ValueError, match=r"Wrong number of inputs"):
        test_model.forward(tuple(torch.randn(8, 10)))
    with pytest.raises(TypeError, match=r"Wrong input type for model! Expected tuple"):
        test_model.forward(torch.randn(8, 10))


# fusilli.fusionmodels.tabularfusion.decision.TabularDecision

def test_TabularDecision():
    test_model = fusion_model_dict["TabularDecision"]

    # attributes available pre-initialisation
    assert hasattr(test_model, "method_name")
    assert test_model.method_name == "Tabular decision"
    assert hasattr(test_model, "modality_type")
    assert test_model.modality_type == "tabular_tabular"
    assert hasattr(test_model, "fusion_type")
    assert test_model.fusion_type == "operation"

    test_model = test_model(prediction_task="binary", data_dims=[10, 14, None], multiclass_dimensions=None)

    # initialising
    assert isinstance(test_model, nn.Module)
    assert isinstance(test_model, ParentFusionModel)
    assert hasattr(test_model, "prediction_task")
    assert test_model.prediction_task == "binary"
    assert hasattr(test_model, "mod1_layers")
    assert test_model.mod1_layers['layer 1'][0].in_features == 10
    assert hasattr(test_model, "mod2_layers")
    assert test_model.mod2_layers['layer 1'][0].in_features == 14
    assert hasattr(test_model, "final_prediction_tab1")
    assert hasattr(test_model, "final_prediction_tab2")
    assert hasattr(test_model, "forward")
    assert hasattr(test_model, "fusion_operation")
    # asserting that the default fusion operation is a mean with a simple calculation
    assert test_model.fusion_operation(torch.randn(8, 10), torch.randn(8, 10)).shape == torch.Size([8, 10])
    assert test_model.fusion_operation(torch.Tensor([1.0]), torch.Tensor([2.0])) == torch.Tensor([1.5])

    # forward pass
    test_input = (torch.randn(8, 10), torch.randn(8, 14))
    test_output = test_model.forward(test_input)
    assert isinstance(test_output, list)
    assert test_output[0].shape == torch.Size([8, 1])
    assert len(test_output) == 1

    # wrong input
    # - too many dimensions
    with pytest.raises(ValueError, match=r"Wrong number of inputs"):
        test_model.forward((torch.randn(8, 10), torch.randn(8, 14), torch.randn(8, 10)))
    # - too few dimensions
    with pytest.raises(ValueError, match=r"Wrong number of inputs"):
        test_model.forward(())
    with pytest.raises(ValueError, match=r"Wrong number of inputs"):
        test_model.forward(tuple(torch.randn(8, 10)))
    with pytest.raises(TypeError, match=r"Wrong input type for model! Expected tuple"):
        test_model.forward(torch.randn(8, 10))


# fusilli.fusionmodels.tabularfusion.mcvae_model.MCVAE_tab
# def test_MCVAE_tab():
#     # just looking at the forward function rather than subspace method too
#     test_model = fusion_model_dict["MCVAE_tab"]
#
#     # attributes available pre-initialisation
#     assert hasattr(test_model, "method_name")
#     assert test_model.method_name == "MCVAE Tabular"
#     assert hasattr(test_model, "modality_type")
#     assert test_model.modality_type == "tabular_tabular"
#     assert hasattr(test_model, "fusion_type")
#     assert test_model.fusion_type == "subspace"
#     assert hasattr(test_model, "subspace_method")
#
#     test_model = test_model(prediction_task="binary", data_dims=[25, None, None], multiclass_dimensions=None)
#
#     # initialising
#     assert isinstance(test_model, nn.Module)
#     assert isinstance(test_model, ParentFusionModel)
#     assert hasattr(test_model, "prediction_task")
#     assert test_model.prediction_task == "binary"
#     assert hasattr(test_model, "latent_space_layers")
#     assert test_model.latent_space_layers['layer 1'][0].in_features == 25
#     assert hasattr(test_model, "fused_dim")
#     assert test_model.fused_dim == test_model.latent_space_layers['layer 5'][0].out_features
#     assert hasattr(test_model, "fused_layers")
#     assert test_model.fused_layers[0].in_features == test_model.latent_space_layers['layer 5'][0].out_features
#     assert hasattr(test_model, "final_prediction")
#     assert hasattr(test_model, "forward")
#
#     # forward pass
#     test_input = torch.randn(8, 25)
#     test_output = test_model.forward(test_input)
#     assert isinstance(test_output, list)
#     assert test_output[0].shape == torch.Size([8, 1])
#     assert len(test_output) == 1
#
#     # wrong input
#     with pytest.raises(TypeError, match=r"Wrong input type for model! Expected torch.Tensor"):
#         test_model.forward([torch.randn(8, 25)])


def test_EdgeCorrGNN():
    test_model = fusion_model_dict["EdgeCorrGNN"]

    # attributes available pre-initialisation
    assert hasattr(test_model, "method_name")
    assert test_model.method_name == "Edge Correlation GNN"
    assert hasattr(test_model, "modality_type")
    assert test_model.modality_type == "tabular_tabular"
    assert hasattr(test_model, "fusion_type")
    assert test_model.fusion_type == "graph"
    assert hasattr(test_model, "graph_maker")

    node_features_dim = 14  # dimensions of modality 2
    num_nodes = 8
    num_edges = 20
    edge_attr_dim = 1  # Adjust as needed

    node_features = torch.randn(num_nodes, node_features_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
    edge_attr = torch.randn(num_edges, edge_attr_dim)
    test_graph_data = (node_features, edge_index, edge_attr)

    test_model = test_model(prediction_task="binary", data_dims=[10, 14, None], multiclass_dimensions=None)

    # initialising
    assert isinstance(test_model, nn.Module)
    assert isinstance(test_model, ParentFusionModel)
    assert hasattr(test_model, "prediction_task")
    assert test_model.prediction_task == "binary"
    assert hasattr(test_model, "graph_conv_layers")
    assert hasattr(test_model, "dropout_prob")
    assert hasattr(test_model, "final_prediction")
    assert hasattr(test_model, "forward")

    # forward pass
    test_output = test_model.forward(test_graph_data)
    assert isinstance(test_output, list)
    assert test_output[0].shape == torch.Size([8, 1])
    assert len(test_output) == 1

    # wrong input
    # type
    with pytest.raises(TypeError, match=r"Wrong input type for model!"):
        test_model.forward([torch.randn(8, 25)])
    # length
    with pytest.raises(ValueError, match=r"Wrong number of inputs for model!"):
        test_model.forward((torch.randn(8, 25), torch.randn(8, 25)))


# fusilli.fusionmodels.tabularfusion.activation.ActivationFusion

def test_ActivationFusion():
    test_model = fusion_model_dict["ActivationFusion"]

    # attributes available pre-initialisation
    assert hasattr(test_model, "method_name")
    assert test_model.method_name == "Activation function map fusion"
    assert hasattr(test_model, "modality_type")
    assert test_model.modality_type == "tabular_tabular"
    assert hasattr(test_model, "fusion_type")
    assert test_model.fusion_type == "operation"

    test_model = test_model(prediction_task="binary", data_dims=[10, 14, None], multiclass_dimensions=None)

    # initialising
    assert isinstance(test_model, nn.Module)
    assert isinstance(test_model, ParentFusionModel)
    assert hasattr(test_model, "prediction_task")
    assert test_model.prediction_task == "binary"
    assert hasattr(test_model, "mod1_layers")
    assert test_model.mod1_layers['layer 1'][0].in_features == 10
    assert hasattr(test_model, "mod2_layers")
    assert test_model.mod2_layers['layer 1'][0].in_features == 14
    assert hasattr(test_model, "fused_dim")
    assert test_model.fused_dim == test_model.mod1_layers['layer 5'][0].out_features + test_model.mod2_layers[
        'layer 5'][0].out_features
    assert hasattr(test_model, "fused_layers")
    assert hasattr(test_model, "final_prediction")
    assert hasattr(test_model, "forward")

    # forward pass
    test_input = (torch.randn(8, 10), torch.randn(8, 14))
    test_output = test_model.forward(test_input)
    assert isinstance(test_output, list)
    assert test_output[0].shape == torch.Size([8, 1])
    assert len(test_output) == 1

    # wrong input
    # - too many dimensions
    with pytest.raises(ValueError, match=r"Wrong number of inputs"):
        test_model.forward((torch.randn(8, 10), torch.randn(8, 14), torch.randn(8, 10)))
    # - too few dimensions
    with pytest.raises(ValueError, match=r"Wrong number of inputs"):
        test_model.forward(())
    with pytest.raises(ValueError, match=r"Wrong number of inputs"):
        test_model.forward(tuple(torch.randn(8, 10)))
    with pytest.raises(TypeError, match=r"Wrong input type for model! Expected tuple"):
        test_model.forward(torch.randn(8, 10))


def test_AttentionAndSelfActivation():
    test_model = fusion_model_dict["AttentionAndSelfActivation"]

    # attributes available pre-initialisation
    assert hasattr(test_model, "method_name")
    assert test_model.method_name == "Activation function and tabular self-attention"
    assert hasattr(test_model, "modality_type")
    assert test_model.modality_type == "tabular_tabular"
    assert hasattr(test_model, "fusion_type")
    assert test_model.fusion_type == "operation"

    test_model = test_model(prediction_task="binary", data_dims=[10, 14, None], multiclass_dimensions=None)

    # initialising
    assert isinstance(test_model, nn.Module)
    assert isinstance(test_model, ParentFusionModel)
    assert hasattr(test_model, "prediction_task")
    assert test_model.prediction_task == "binary"
    assert hasattr(test_model, "mod1_layers")
    assert test_model.mod1_layers['layer 1'][0].in_features == 10
    assert hasattr(test_model, "mod2_layers")
    assert test_model.mod2_layers['layer 1'][0].in_features == 14
    assert hasattr(test_model, "fused_dim")
    assert test_model.fused_dim == test_model.mod1_layers['layer 5'][0].out_features + test_model.mod2_layers[
        'layer 5'][0].out_features
    assert hasattr(test_model, "fused_layers")
    assert hasattr(test_model, "final_prediction")
    assert hasattr(test_model, "forward")

    # forward pass
    test_input = (torch.randn(8, 10), torch.randn(8, 14))
    test_model.attention_reduction_ratio = 2
    test_output = test_model.forward(test_input)
    assert isinstance(test_output, list)
    assert test_output[0].shape == torch.Size([8, 1])
    assert len(test_output) == 1

    # wrong input
    # - too many dimensions
    with pytest.raises(ValueError, match=r"Wrong number of inputs"):
        test_model.forward((torch.randn(8, 10), torch.randn(8, 14), torch.randn(8, 10)))
    # - too few dimensions
    with pytest.raises(ValueError, match=r"Wrong number of inputs"):
        test_model.forward(())
    with pytest.raises(ValueError, match=r"Wrong number of inputs"):
        test_model.forward(tuple(torch.randn(8, 10)))
    with pytest.raises(TypeError, match=r"Wrong input type for model! Expected tuple"):
        test_model.forward(torch.randn(8, 10))
    with pytest.raises(UserWarning, match=r"first tabular modality dimensions // attention_reduction_ratio < 1"):
        test_model.attention_reduction_ratio = 16
        test_model.forward((torch.randn(8, 10), torch.randn(8, 14)))


def test_AttentionWeightedGNN():
    test_model = fusion_model_dict["AttentionWeightedGNN"]

    # attributes available pre-initialisation
    assert hasattr(test_model, "method_name")
    assert test_model.method_name == "Attention-weighted GNN"
    assert hasattr(test_model, "modality_type")
    assert test_model.modality_type == "tabular_tabular"
    assert hasattr(test_model, "fusion_type")
    assert test_model.fusion_type == "graph"
    assert hasattr(test_model, "graph_maker")

    node_features_dim = 14  # dimensions of modality 2
    num_nodes = 8
    num_edges = 20
    edge_attr_dim = 1  # Adjust as needed

    node_features = torch.randn(num_nodes, node_features_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
    edge_attr = torch.randn(num_edges, edge_attr_dim)
    test_graph_data = (node_features, edge_index, edge_attr)

    test_model = test_model(prediction_task="binary", data_dims=[10, 14, None], multiclass_dimensions=None)

    # initialising
    assert isinstance(test_model, nn.Module)
    assert isinstance(test_model, ParentFusionModel)
    assert hasattr(test_model, "prediction_task")
    assert test_model.prediction_task == "binary"
    assert hasattr(test_model, "graph_conv_layers")
    assert hasattr(test_model, "dropout_prob")
    assert hasattr(test_model, "final_prediction")
    assert hasattr(test_model, "forward")

    # forward pass
    test_output = test_model.forward(test_graph_data)
    assert isinstance(test_output, list)
    assert test_output[0].shape == torch.Size([8, 1])
    assert len(test_output) == 1

    # wrong input
    # type
    with pytest.raises(TypeError, match=r"Wrong input type for model!"):
        test_model.forward([torch.randn(8, 25)])
    # length
    with pytest.raises(ValueError, match=r"Wrong number of inputs for model!"):
        test_model.forward((torch.randn(8, 25), torch.randn(8, 25)))
