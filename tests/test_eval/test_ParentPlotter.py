import pytest
import torch
from fusilli.eval import ParentPlotter, RealsVsPreds, ConfusionMatrix, ModelComparison
from unittest.mock import Mock
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import numpy as np
import torchmetrics as tm


class ModelParams:
    def __init__(self):
        self.params = {}
        self.method_name = "test"


class ExampleModel:
    def __init__(self):
        self.model = ModelParams()

        self.metrics = {
            "regression": [
                {"metric": tm.R2Score(), "name": "R2"},
                {"metric": tm.MeanAbsoluteError(), "name": "MAE"},
            ],
        }

    def set_model_params(self, params):
        self.model.params = params
        self.model.pred_type = params["pred_type"]

    def set_params(self, params):
        for key, value in params.items():
            setattr(self, key, value)

    def get_params(self):
        return self.model.params


@pytest.fixture
def sample_kfold_model_data():
    # Define sample k-fold model data for testing
    model = ExampleModel()
    model.set_model_params(
        {
            "kfold_flag": True,
            "pred_type": "binary",
            "num_k": 2,
        }
    )

    model_data = {
        "pred_type": "binary",
        "train_reals": [torch.tensor([1, 0, 1]), torch.tensor([0, 1, 0])],
        "train_preds": [torch.tensor([1, 0, 1]), torch.tensor([0, 1, 0])],
        "val_reals": [torch.tensor([1, 0, 1]), torch.tensor([0, 1, 0])],
        "val_preds": [torch.tensor([1, 0, 1]), torch.tensor([0, 1, 0])],
        "metrics_per_fold": {"metric1": [0.85, 0.78], "metric2": [0.92, 0.88]},
        "overall_kfold_metrics": {"metric1": 0.85, "metric2": 0.92},
    }

    model.set_params(model_data.copy())

    return model


@pytest.fixture
def sample_train_test_model_data():
    # Define sample train/test model data for testing

    model = ExampleModel()
    model.set_model_params(
        {
            "kfold_flag": False,
            "pred_type": "binary",
        }
    )

    model_data = {
        "pred_type": "binary",
        "train_reals": torch.tensor([1, 0, 1]),
        "train_preds": torch.tensor([1, 0, 1]),
        "val_reals": torch.tensor([1, 0, 1]),
        "val_preds": torch.tensor([1, 0, 1]),
        "metric_values": {"metric1": 0.85, "metric2": 0.92},
    }

    model.set_params(model_data.copy())

    return model


def test_from_final_val_data_invalid_input():
    with pytest.raises(ValueError) as excinfo:
        # Pass an empty list as input, which is invalid
        RealsVsPreds.from_final_val_data([])

    assert "Argument 'model_list' is an empty list" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        # Pass a list with no models as input, which is invalid
        RealsVsPreds.from_final_val_data({})

    assert "Argument 'model_list' is not a list" in str(excinfo.value)

    # Confusion Matrix
    with pytest.raises(ValueError) as excinfo:
        # Pass an empty list as input, which is invalid
        ConfusionMatrix.from_final_val_data([])
    assert "Argument 'model_list' is an empty list" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        # Pass a list with no models as input, which is invalid
        ConfusionMatrix.from_final_val_data({})
    assert "Argument 'model_list' is not a list" in str(excinfo.value)


# Test case for calling get_new_kfold_data when kfold flag is True and model_list has more than 1 entry
def test_get_kfold_data_from_model_called(sample_kfold_model_data, mocker):
    # Mock the get_kfold_data_from_model method to check if it's called

    mocker.patch.object(
        ParentPlotter,
        "get_kfold_data_from_model",
        return_value=(
            sample_kfold_model_data.train_reals,
            sample_kfold_model_data.train_preds,
            sample_kfold_model_data.val_reals,
            sample_kfold_model_data.val_preds,
            sample_kfold_model_data.metrics_per_fold,
            sample_kfold_model_data.overall_kfold_metrics,
        ),
    )

    # Create a sample model_list with kfold_flag as True
    model_list = [sample_kfold_model_data, sample_kfold_model_data]

    # Call the method
    RealsVsPreds.from_final_val_data(model_list)
    ParentPlotter.get_kfold_data_from_model.assert_called()

    # Confusion Matrix
    ConfusionMatrix.from_final_val_data(model_list)
    ParentPlotter.get_kfold_data_from_model.assert_called()

    # Assert called twice
    assert ParentPlotter.get_kfold_data_from_model.call_count == 2


def test_get_new_kfold_data_called(sample_kfold_model_data, mocker):
    # Mock the get_kfold_data_from_model method to check if it's called

    mocker.patch.object(
        ParentPlotter,
        "get_new_kfold_data",
        return_value=(
            sample_kfold_model_data.train_reals,
            sample_kfold_model_data.train_preds,
            sample_kfold_model_data.val_reals,
            sample_kfold_model_data.val_preds,
            sample_kfold_model_data.metrics_per_fold,
            sample_kfold_model_data.overall_kfold_metrics,
        ),
    )

    # Create a sample model_list with kfold_flag as True
    model_list = [sample_kfold_model_data, sample_kfold_model_data]

    # Call the method
    RealsVsPreds.from_new_data(model_list, sample_kfold_model_data.val_reals[0])
    ParentPlotter.get_new_kfold_data.assert_called()

    # Confusion Matrix
    ConfusionMatrix.from_new_data(model_list, sample_kfold_model_data.val_reals[0])
    ParentPlotter.get_new_kfold_data.assert_called()

    assert ParentPlotter.get_new_kfold_data.call_count == 2


# Test case for an error when model_list has more than 1 entry but kfold is False
def test_error_when_kfold_false_with_multiple_models(sample_train_test_model_data):
    # Create a sample model_list with kfold_flag as False
    model_list = [sample_train_test_model_data, sample_train_test_model_data]

    # Use pytest.raises to check if a ValueError is raised
    with pytest.raises(ValueError, match="kfold_flag is False"):
        RealsVsPreds.from_final_val_data(model_list)

    # Confusion Matrix
    with pytest.raises(ValueError, match="kfold_flag is False"):
        ConfusionMatrix.from_final_val_data(model_list)


# Test case for calling get_new_tt_data when len(model) == 1
def test_get_tt_data_from_model_called(sample_train_test_model_data, mocker):
    # Mock the get_tt_data_from_model method to check if it's called
    mocker.patch.object(
        ParentPlotter,
        "get_tt_data_from_model",
        return_value=(
            sample_train_test_model_data.train_reals,
            sample_train_test_model_data.train_preds,
            sample_train_test_model_data.val_reals,
            sample_train_test_model_data.val_preds,
            sample_train_test_model_data.metric_values,
        ),
    )

    # Create a sample model_list with a single model
    model_list = [sample_train_test_model_data]

    # Call the method
    RealsVsPreds.from_final_val_data(model_list)
    ParentPlotter.get_tt_data_from_model.assert_called()

    # Confusion Matrix
    ConfusionMatrix.from_final_val_data(model_list)
    ParentPlotter.get_tt_data_from_model.assert_called()

    # Assert called twice
    assert ParentPlotter.get_tt_data_from_model.call_count == 2


# Test case for calling get_new_tt_data when len(model) == 1
def test_get_new_tt_data_called(sample_train_test_model_data, mocker):
    # Mock the get_tt_data_from_model method to check if it's called
    mocker.patch.object(
        ParentPlotter,
        "get_new_tt_data",
        return_value=(
            sample_train_test_model_data.train_reals,
            sample_train_test_model_data.train_preds,
            sample_train_test_model_data.val_reals,
            sample_train_test_model_data.val_preds,
            sample_train_test_model_data.metric_values,
        ),
    )

    # Create a sample model_list with a single model
    model_list = [sample_train_test_model_data]

    # Call the method
    RealsVsPreds.from_new_data(model_list, sample_train_test_model_data.val_reals[0])
    ParentPlotter.get_new_tt_data.assert_called()

    # Confusion Matrix
    ConfusionMatrix.from_new_data(model_list, sample_train_test_model_data.val_reals[0])
    ParentPlotter.get_new_tt_data.assert_called()

    # Assert called twice
    assert ParentPlotter.get_new_tt_data.call_count == 2


# Test case for an error when model_list has length 1 but kfold flag is true
def test_error_when_kfold_true_with_single_model(sample_kfold_model_data):
    # Create a sample model_list with kfold_flag as True
    model_list = [sample_kfold_model_data]

    # Use pytest.raises to check if a ValueError is raised
    with pytest.raises(ValueError, match="kfold_flag is True"):
        RealsVsPreds.from_final_val_data(model_list)

    # confusion matrix
    with pytest.raises(ValueError, match="kfold_flag is True"):
        ConfusionMatrix.from_final_val_data(model_list)


# Model comparison
