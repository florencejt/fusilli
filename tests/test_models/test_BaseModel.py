import re

import pytest
from fusilli.fusionmodels.base_model import BaseModel
import torch
import torch.nn as nn
from unittest.mock import Mock


class SampleGraphFusionModel(nn.Module):
    def __init__(self, prediction_task, multiclass_dimensions=3):
        super(SampleGraphFusionModel, self).__init__()
        self.prediction_task = prediction_task
        self.multiclass_dimensions = multiclass_dimensions
        self.fusion_type = "graph"  # Sample fusion_type
        self.subspace_method = None
        self.graph_maker = Mock()

    def forward(self, x):
        return [torch.rand((x[0].shape[0], 1)), ]


class SampleFusionModel(nn.Module):
    def __init__(self, prediction_task, multiclass_dimensions=3):
        super(SampleFusionModel, self).__init__()
        self.prediction_task = prediction_task
        self.multiclass_dimensions = multiclass_dimensions
        self.fusion_type = "attention"  # Sample fusion_type
        self.subspace_method = None

    def forward(self, x):
        if isinstance(x, tuple):
            return [torch.rand((x[0].shape[0], 1)), ]
        else:
            return [torch.rand((x.shape[0], 1)), ]


class SampleFusionModelReconstructions(nn.Module):
    def __init__(self, prediction_task, multiclass_dimensions=3):
        super(SampleFusionModelReconstructions, self).__init__()
        self.prediction_task = prediction_task
        self.multiclass_dimensions = multiclass_dimensions
        self.fusion_type = "attention"  # Sample fusion_type
        self.subspace_method = None
        self.custom_loss = Mock(return_value=0.5)

    def forward(self, x):
        if isinstance(x, tuple):
            return [torch.rand((x[0].shape[0], 1)), torch.rand(x[0].shape)]
        else:
            return [torch.rand((x.shape[0], 1)), torch.rand(x.shape)]


@pytest.fixture
def sample_graph_model():
    return BaseModel(SampleGraphFusionModel(prediction_task="binary"))


@pytest.fixture
def sample_model():
    return BaseModel(SampleFusionModel(prediction_task="binary"))


@pytest.fixture
def sample_model_recon():
    return BaseModel(SampleFusionModelReconstructions(prediction_task="binary"))


def test_safe_squeeze(sample_model):
    model = sample_model

    # Test for 1D tensor
    tensor = torch.Tensor([1, 2, 3])
    squeezed = model.safe_squeeze(tensor)
    assert squeezed.shape == torch.Size([3])

    # Test for 2D tensor with 1 row
    tensor = torch.Tensor([[1, 2, 3]])
    squeezed = model.safe_squeeze(tensor)
    assert squeezed.shape == torch.Size([3])

    # Test for 2D tensor with 1 column
    tensor = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    squeezed = model.safe_squeeze(tensor)
    assert squeezed.shape == torch.Size([2, 3])


def test_get_data_from_batch(sample_model):
    model = sample_model

    # uni-modal
    batch = (torch.rand((10, 12)), torch.randint(1, (10,)))
    x, y = model.get_data_from_batch(batch)

    assert isinstance(x, torch.Tensor)
    assert x.shape == torch.Size([10, 12])
    assert isinstance(y, torch.Tensor)
    assert y.shape == torch.Size([10])

    # multi-modal: 2 modalities, dims are 12 features and 15 features
    batch = (torch.rand((10, 12)), torch.rand((10, 15)), torch.randint(1, (10,)))
    x, y = model.get_data_from_batch(batch)

    assert isinstance(x, tuple)
    assert len(x) == 2
    assert isinstance(x[0], torch.Tensor)
    assert x[0].shape == torch.Size([10, 12])
    assert isinstance(x[1], torch.Tensor)
    assert x[1].shape == torch.Size([10, 15])
    assert isinstance(y, torch.Tensor)
    assert y.shape == torch.Size([10])

    # more than 2 modalities
    batch = (torch.rand((10, 12)), torch.rand((10, 15)), torch.rand((10, 18)), torch.randint(1, (10,)))
    with pytest.raises(ValueError, match=re.escape("Batch size is not 2 (preds and labels) or 3 (2 pred data types "
                                                   "and "
                                                   "labels) modalities long")):
        x, y = model.get_data_from_batch(batch)


def test_get_data_from_batch_graph(sample_graph_model):
    model = sample_graph_model

    class GraphDataClass():
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.edge_index = torch.tensor([[0, 1, 1, 2],
                                            [1, 0, 2, 1]], dtype=torch.long)
            self.edge_attr = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float)

    batch = GraphDataClass(torch.rand((12, 10)), torch.randint(1, (10,)))

    x, y = model.get_data_from_batch(batch)

    assert isinstance(x, tuple)
    assert len(x) == 3
    assert isinstance(x[0], torch.Tensor)
    assert x[0].shape == torch.Size([12, 10])
    assert isinstance(x[1], torch.Tensor)
    assert x[1].shape == torch.Size([2, 4])
    assert isinstance(x[2], torch.Tensor)
    assert x[2].shape == torch.Size([4, 1])
    assert isinstance(y, torch.Tensor)
    assert y.shape == torch.Size([10])


# get_model_outputs
def test_get_model_outputs(sample_model, sample_model_recon):
    model = sample_model

    # no reconstructions
    # model.model = Mock(return_value=[torch.rand((10, 1)), ])
    batch = (torch.rand((10, 12)), torch.randint(1, (10,)))
    preds, reconstructions = model.get_model_outputs(batch)
    assert isinstance(preds, torch.Tensor)
    assert preds.shape == torch.Size([10])
    assert isinstance(reconstructions, list)
    assert len(reconstructions) == 0

    # with reconstructions
    model2 = sample_model_recon
    preds, reconstructions = model2.get_model_outputs(batch)

    assert isinstance(preds, torch.Tensor)
    assert preds.shape == torch.Size([10])
    assert isinstance(reconstructions, list)
    assert len(reconstructions) == 1
    assert isinstance(reconstructions[0], torch.Tensor)
    assert reconstructions[0].shape == torch.Size([10, 12])


# get model outputs and loss
def test_get_model_outputs_and_loss(sample_model, sample_model_recon):
    model = sample_model
    x = torch.rand((10, 12))
    y = torch.randint(1, (10,))

    loss, end_output, logits = model.get_model_outputs_and_loss(x, y)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])  # scalar
    assert isinstance(end_output, torch.Tensor)
    assert end_output.shape == torch.Size([1, 10])
    assert torch.all(torch.logical_or(end_output == 0, end_output == 1))
    assert torch.all(torch.logical_and(logits >= 0, logits <= 1))
    assert logits.shape == torch.Size([1, 10])

    # with reconstructions
    model2 = sample_model_recon
    x = torch.rand((10, 12))
    y = torch.randint(1, (10,))

    loss, end_output, logits = model2.get_model_outputs_and_loss(x, y)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])  # scalar
    assert isinstance(end_output, torch.Tensor)
    assert end_output.shape == torch.Size([1, 10])
    assert torch.all(torch.logical_or(end_output == 0, end_output == 1))
    assert torch.all(torch.logical_and(logits >= 0, logits <= 1))
    assert logits.shape == torch.Size([1, 10])
    model2.model.custom_loss.assert_called_once()


# get_model_outputs and loss with graph fusion
def test_get_model_outputs_and_loss_graph(sample_graph_model):
    model = sample_graph_model

    model.train_mask = torch.tensor([True, True, True, True, True, True, True, False, False, False], dtype=torch.bool)
    model.val_mask = torch.tensor([False, False, False, False, False, False, False, True, True, True], dtype=torch.bool)

    x = (torch.rand((10, 12)), torch.tensor([[0, 1, 1, 2],
                                             [1, 0, 2, 1]], dtype=torch.long),
         torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float))

    y = torch.randint(1, (10,))

    loss, end_output, logits = model.get_model_outputs_and_loss(x, y, train=True)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])  # scalar
    assert isinstance(end_output, torch.Tensor)
    assert end_output.shape == torch.Size([7])
    assert torch.all(torch.logical_or(end_output == 0, end_output == 1))
    assert torch.all(torch.logical_and(logits >= 0, logits <= 1))
    assert logits.shape == torch.Size([7])

    # train = False
    loss, end_output, logits = model.get_model_outputs_and_loss(x, y, train=False)

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])  # scalar
    assert isinstance(end_output, torch.Tensor)
    assert end_output.shape == torch.Size([3])
    assert torch.all(torch.logical_or(end_output == 0, end_output == 1))
    assert torch.all(torch.logical_and(logits >= 0, logits <= 1))
    assert logits.shape == torch.Size([3])


def test_metrics_exist(sample_model):
    model = sample_model

    assert isinstance(model.metrics, dict)
    assert len(model.metrics) >= 2


@pytest.mark.filterwarnings("ignore:.*You are trying to `self.log()`*.",
                            "ignore:.*No positive samples in targets*.")
def test_training_step(sample_model):
    model = sample_model
    batch = (torch.rand((10, 10)), torch.randint(1, (10,)))
    loss = model.training_step(batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])  # scalar


@pytest.mark.filterwarnings("ignore:.*You are trying to `self.log()`*.")
def test_validation_step(sample_model):
    model = sample_model
    batch = (torch.rand((10, 10)), torch.randint(2, (10,)))
    model.validation_step(batch, batch_idx=0)


@pytest.mark.filterwarnings("ignore:.*You are trying to `self.log()`*.")
def test_predict_step(sample_model):
    model = sample_model
    batch = (torch.rand((10, 10)), torch.randint(2, (10,)))
    end_output, logits = model.predict_step(batch, batch_idx=0)
    assert isinstance(end_output, torch.Tensor)
    assert isinstance(logits, torch.Tensor)
