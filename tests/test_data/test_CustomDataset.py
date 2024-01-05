import pytest
import pandas as pd
import torch
from fusilli.data import CustomDataset  # Import your CustomDataset class


# Test initialization with multimodal data (list of tensors)
def test_init_multimodal_data():
    data1 = torch.randn(10, 3, 32, 32)
    data2 = torch.randn(10, 5)
    labels = pd.DataFrame({"prediction_label": [0] * 10})
    dataset = CustomDataset([data1, data2], labels)
    assert dataset.multimodal_flag
    assert len(dataset) == 10


# Test initialization with unimodal data (single tensor)
def test_init_unimodal_data():
    data = torch.randn(10, 3, 32, 32)
    labels = pd.DataFrame({"prediction_label": [0] * 10})
    dataset = CustomDataset(data, labels)
    assert not dataset.multimodal_flag
    assert len(dataset) == 10


# Test initialization with invalid pred_features type
def test_init_invalid_pred_features_type():
    invalid_data = "invalid_data"
    labels = pd.DataFrame({"prediction_label": [0] * 10})

    with pytest.raises(ValueError):
        CustomDataset(invalid_data, labels)


# Test label conversion to long
def test_label_conversion_to_long():
    data = torch.randn(10, 3, 32, 32)
    labels = pd.DataFrame({"prediction_label": [0] * 10})
    dataset = CustomDataset(data, labels)
    assert dataset.labels.dtype == torch.int64


# Test label conversion to float
def test_label_conversion_to_float():
    data = torch.randn(10, 3, 32, 32)
    labels = pd.DataFrame({"prediction_label": [0.0] * 10})
    dataset = CustomDataset(data, labels)
    assert dataset.labels.dtype == torch.float32


# Test __getitem__ method for multimodal data
def test_getitem_multimodal_data():
    data1 = torch.randn(10, 3, 32, 32)
    data2 = torch.randn(10, 5)
    labels = pd.DataFrame({"prediction_label": [0] * 10})
    dataset = CustomDataset([data1, data2], labels)

    sample = dataset[0]
    assert len(sample) == 3  # Should return a tuple of 3 elements


# Test __getitem__ method for unimodal data
def test_getitem_unimodal_data():
    data = torch.randn(10, 3, 32, 32)
    labels = pd.DataFrame({"prediction_label": [0] * 10})
    dataset = CustomDataset(data, labels)

    sample = dataset[0]
    assert len(sample) == 2  # Should return a tuple of 2 elements


# Test __getitem__ method with invalid index
def test_getitem_invalid_index():
    data = torch.randn(10, 3, 32, 32)
    labels = pd.DataFrame({"prediction_label": [0] * 10})
    dataset = CustomDataset(data, labels)

    with pytest.raises(IndexError):
        dataset[20]  # Index out of range


# Run pytest
if __name__ == "__main__":
    pytest.main()
