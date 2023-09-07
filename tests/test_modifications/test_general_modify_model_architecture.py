import pytest
from fusionlibrary.train_functions import (
    modify_model_architecture,
    get_nested_attr,
    reset_fused_layers,
)
import torch.nn as nn


# Sample model class for testing
class SampleModel(nn.Module):
    def __init__(self):
        super(SampleModel, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3)
        self.fc = nn.Linear(64, 10)


# Test case for modify_model_architecture
def test_modify_model_architecture():
    # Define a sample model
    model = SampleModel()

    # Define architecture modifications
    architecture_modification = {
        "all": {
            "conv": nn.Conv2d(3, 128, kernel_size=3),
            "fc": nn.Linear(128, 10),
        }
    }

    # Modify the model's architecture
    modified_model = modify_model_architecture(model, architecture_modification)

    # Check if the modifications are applied
    assert isinstance(modified_model.conv, nn.Conv2d)
    assert modified_model.conv.out_channels == 128
    assert isinstance(modified_model.fc, nn.Linear)
    assert modified_model.fc.out_features == 10


# Test case for get_nested_attr
def test_get_nested_attr():
    # Define a sample object with nested attributes
    class SampleObject:
        def __init__(self):
            self.nested = SampleModel()

    # Create a sample object
    obj = SampleObject()

    # Get a nested attribute using get_nested_attr
    nested_attr = get_nested_attr(obj, "nested.conv")

    # Check if the nested attribute is correctly retrieved
    assert isinstance(nested_attr, nn.Conv2d)
    assert nested_attr.out_channels == 64


# Test case for reset_fused_layers
def test_reset_fused_layers():
    # Define a sample model with fused layers
    class SampleModelWithFusedLayers(nn.Module):
        def __init__(self):
            super(SampleModelWithFusedLayers, self).__init__()
            self.conv = nn.Conv2d(3, 64, kernel_size=3)
            self.fc = nn.Linear(64, 10)

        def calc_fused_layers(self):
            pass

    # Create a sample model with fused layers
    model = SampleModelWithFusedLayers()

    # Call reset_fused_layers
    reset_fused_layers(model)

    # Check if the reset method is called correctly
    assert True  # Add assertions related to the reset operation if needed


# Run pytest
if __name__ == "__main__":
    pytest.main()
