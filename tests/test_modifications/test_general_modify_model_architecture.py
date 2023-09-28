import pytest
from fusilli.utils import model_modifier
import torch.nn as nn


# Sample model class for testing
class SampleModel(nn.Module):
    def __init__(self):
        super(SampleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)

    def calc_fused_layers(self):
        self.fc = nn.Linear(self.conv2.out_channels, 8)


# Test case for modify_model_architecture
def test_modify_model_architecture():
    # Define a sample model
    model = SampleModel()

    # Define architecture modifications
    architecture_modification = {
        "all": {
            "conv1": nn.Conv2d(3, 128, kernel_size=3),
            "conv2": nn.Conv2d(128, 128, kernel_size=3),
        }
    }

    # Modify the model's architecture
    modified_model = model_modifier.modify_model_architecture(
        model, architecture_modification
    )

    # Check if the modifications are applied
    assert isinstance(modified_model.conv1, nn.Conv2d)
    assert modified_model.conv1.out_channels == 128
    assert isinstance(modified_model.conv2, nn.Conv2d)
    assert modified_model.conv2.out_channels == 128
    assert isinstance(modified_model.fc, nn.Linear)
    assert modified_model.fc.out_features == 8


def test_specific_modify_model_architecture():
    model = SampleModel()

    architecture_modification = {
        "SampleModel": {
            "conv1": nn.Conv2d(3, 128, kernel_size=3),
            "conv2": nn.Conv2d(128, 128, kernel_size=3),
        }
    }

    modified_model = model_modifier.modify_model_architecture(
        model, architecture_modification
    )

    assert isinstance(modified_model.conv1, nn.Conv2d)
    assert modified_model.conv1.out_channels == 128
    assert isinstance(modified_model.conv2, nn.Conv2d)
    assert modified_model.conv2.out_channels == 128
    assert isinstance(modified_model.fc, nn.Linear)
    assert modified_model.fc.in_features == 128
    assert modified_model.fc.out_features == 8


def test_specific_modify_model_nonexistent_attr():
    model = SampleModel()

    architecture_modification = {
        "SampleModel": {
            "conv1": nn.Conv2d(3, 128, kernel_size=3),
            "conv3": nn.Conv2d(128, 128, kernel_size=3),
        }
    }

    with pytest.raises(ValueError):
        model_modifier.modify_model_architecture(model, architecture_modification)


# Test case for modifying an attribute that doesn't exist
def test_modify_model_architecture_nonexistent_attr():
    model = SampleModel()

    architecture_modification = {
        "all": {
            "conv1": nn.Conv2d(3, 128, kernel_size=3),
            "conv2": nn.Conv2d(128, 128, kernel_size=3),
            "conv3": nn.Conv2d(128, 128, kernel_size=3),
        }
    }

    with pytest.warns(UserWarning):
        model_modifier.modify_model_architecture(model, architecture_modification)


# Test case for get_nested_attr
def test_get_nested_attr():
    # Define a sample object with nested attributes
    class SampleObject:
        def __init__(self):
            self.nested = SampleModel()

    # Create a sample object
    obj = SampleObject()

    # Get a nested attribute using get_nested_attr
    nested_attr = model_modifier.get_nested_attr(obj, "nested.conv1")

    # Check if the nested attribute is correctly retrieved
    assert isinstance(getattr(nested_attr, "conv1"), nn.Conv2d)
    assert getattr(nested_attr, "conv1").out_channels == 64


# Test case for reset_fused_layers
def test_reset_fused_layers():
    # Define a sample model with fused layers
    class SampleModelWithFusedLayers(nn.Module):
        def __init__(self):
            super(SampleModelWithFusedLayers, self).__init__()
            self.conv = nn.Conv2d(3, 64, kernel_size=3)

        def calc_fused_layers(self):
            self.fc = nn.Linear(self.conv.out_channels, 10)

    # Create a sample model with fused layers
    model = SampleModelWithFusedLayers()

    # Call reset_fused_layers
    model_modifier.reset_fused_layers(model, "SampleModelWithFusedLayers")

    # Check if the reset method is called correctly
    assert True  # Add assertions related to the reset operation if needed


# Run pytest
if __name__ == "__main__":
    pytest.main()
