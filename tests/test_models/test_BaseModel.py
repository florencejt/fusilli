import re

import pytest
from fusilli.fusionmodels.base_model import BaseModel
import torch
import torch.nn as nn
from unittest.mock import Mock


class SampleGraphFusionModel(nn.Module):
    def __init__(self, pred_type, multiclass_dim=3):
        super(SampleGraphFusionModel, self).__init__()
        self.pred_type = pred_type
        self.multiclass_dim = multiclass_dim
        self.fusion_type = "graph"  # Sample fusion_type
        self.subspace_method = None
        self.graph_maker = Mock()

    def forward(self, x):
        return [torch.rand((x.shape[0], 1)), ]


class SampleFusionModel(nn.Module):
    def __init__(self, pred_type, multiclass_dim=3):
        super(SampleFusionModel, self).__init__()
        self.pred_type = pred_type
        self.multiclass_dim = multiclass_dim
        self.fusion_type = "attention"  # Sample fusion_type
        self.subspace_method = None

    def forward(self, x):
        if isinstance(x, tuple):
            return [torch.rand((x[0].shape[0], 1)), ]
        else:
            return [torch.rand((x.shape[0], 1)), ]


@pytest.fixture
def sample_graph_model():
    return BaseModel(SampleGraphFusionModel(pred_type="binary"))


@pytest.fixture
def sample_model():
    return BaseModel(SampleFusionModel(pred_type="binary"))


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
    batch = (torch.rand((12, 10)), torch.randint(1, (10,)))
    x, y = model.get_data_from_batch(batch)

    assert isinstance(x, torch.Tensor)
    assert x.shape == torch.Size([12, 10])
    assert isinstance(y, torch.Tensor)
    assert y.shape == torch.Size([10])

    # multi-modal
    batch = (torch.rand((12, 10)), torch.rand((15, 10)), torch.randint(1, (10,)))
    x, y = model.get_data_from_batch(batch)

    assert isinstance(x, tuple)
    assert len(x) == 2
    assert isinstance(x[0], torch.Tensor)
    assert x[0].shape == torch.Size([12, 10])
    assert isinstance(x[1], torch.Tensor)
    assert x[1].shape == torch.Size([15, 10])
    assert isinstance(y, torch.Tensor)
    assert y.shape == torch.Size([10])

    # more than 2 modalities
    batch = (torch.rand((12, 10)), torch.rand((15, 10)), torch.rand((15, 10)), torch.randint(1, (10,)))
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
def test_get_model_outputs(sample_model):
    model = sample_model

    # no reconstructions
    # model.model = Mock(return_value=[torch.rand((10, 1)), ])
    batch = (torch.rand((12, 10)), torch.randint(1, (10,)))
    preds, reconstructions = model.get_model_outputs(batch)
    assert isinstance(preds, torch.Tensor)
    assert preds.shape == torch.Size([10])
    assert reconstructions is None
