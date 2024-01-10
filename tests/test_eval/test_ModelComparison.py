import pytest
from fusilli.eval import ModelComparison, ParentPlotter
import torch
from unittest.mock import Mock
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import numpy as np
import torchmetrics as tm
from pytest_mock import mocker

from .test_ParentPlotter import (
    ExampleModel,
    ModelParams,
    sample_kfold_model_data,
    sample_train_test_model_data,
)


# check it throws an error if the lists are empty
def test_error_when_lists_are_empty():
    model_dict = {
        "model1": [],
        "model2": [],
    }

    with pytest.raises(
            ValueError,
            match="Empty list of models has been passed into the ModelComparison.from_final_val_data.",
    ):
        ModelComparison.from_final_val_data(model_dict)

    with pytest.raises(
            ValueError,
            match="Empty list of models has been passed into the ModelComparison.from_new_data.",
    ):
        ModelComparison.from_new_data(model_dict, output_paths={}, test_data_paths={})


# check it throws an error if model_dict input is not a dict
def test_error_when_model_dict_not_dict():
    with pytest.raises(ValueError, match="Argument 'model_dict' is not a dict"):
        ModelComparison.from_final_val_data("model_dict")
    with pytest.raises(ValueError, match="Argument 'model_dict' is not a dict"):
        ModelComparison.from_new_data("model_dict", output_paths={}, test_data_paths={})


# check it calls get_kfold_data_from_model if kfold is true
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
    model_dict = {
        "model1": [sample_kfold_model_data, sample_kfold_model_data],
        "model2": [sample_kfold_model_data, sample_kfold_model_data],
    }

    # Call the method
    ModelComparison.from_final_val_data(model_dict)
    ParentPlotter.get_kfold_data_from_model.assert_called()

    # Assert called number of times equal to number of models being compared
    assert ModelComparison.get_kfold_data_from_model.call_count == 2


# check it calls get_tt_data_from_model if kfold is false
def test_get_tt_data_from_model_called(sample_train_test_model_data, mocker):
    # Mock the get_kfold_data_from_model method to check if it's called

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

    # Create a sample model_list with kfold_flag as True
    model_dict = {
        "model1": [sample_train_test_model_data],
        "model2": [sample_train_test_model_data],
    }

    # Call the method
    ModelComparison.from_final_val_data(model_dict)
    ParentPlotter.get_tt_data_from_model.assert_called()

    # Assert called number of times equal to number of models being compared
    assert ModelComparison.get_tt_data_from_model.call_count == 2


# ~~~~~~ get_performance_dataframe ~~~~~~


def test_get_performance_dataframe_kfold():
    comparing_models_metrics = {
        "Model1": {"Metric1": [1, 2, 3], "Metric2": [4, 5, 6]},
        "Model2": {"Metric1": [7, 8, 9], "Metric2": [10, 11, 12]},
    }

    overall_kfold_metrics_dict = {
        "Model1": {"Metric1": 2.0, "Metric2": 5.0},
        "Model2": {"Metric1": 8.0, "Metric2": 11.0},
    }

    kfold_flag = True

    # model_comparison = ModelComparison()
    df = ModelComparison.get_performance_dataframe(
        comparing_models_metrics, overall_kfold_metrics_dict, kfold_flag
    )

    # Check that the resulting DataFrame has the expected structure
    assert isinstance(df, pd.DataFrame)
    print(df)
    print(df.index.name)
    # assert "Method" in df.index.name
    assert "Metric1" in df.columns
    assert "Metric2" in df.columns
    assert "fold1_Metric1" in df.columns
    assert "fold1_Metric2" in df.columns

    # Check that the values in the DataFrame match the input data
    assert df.loc["Model1", "Metric1"] == 2.0
    assert df.loc["Model1", "fold1_Metric1"] == 1
    assert df.loc["Model2", "Metric2"] == 11.0
    assert df.loc["Model2", "fold1_Metric2"] == 10


def test_get_performance_dataframe_train_test():
    # Test when kfold_flag is False

    comparing_models_metrics = {
        "Model1": {"Metric1": torch.tensor(1.0), "Metric2": torch.tensor(4.0)},
        "Model2": {"Metric1": torch.tensor(2.0), "Metric2": torch.tensor(5.0)},
    }

    kfold_flag = False

    # model_comparison = ModelComparison()
    df = ModelComparison.get_performance_dataframe(
        comparing_models_metrics, None, kfold_flag
    )

    # Check that the resulting DataFrame has the expected structure
    assert isinstance(df, pd.DataFrame)
    assert "Method" in df.index.name
    assert "Metric1" in df.columns  # because of the .lower()s
    assert "Metric2" in df.columns

    # Check that the values in the DataFrame match the input data
    assert df.loc["Model1", "Metric1"] == 1.0
    assert df.loc["Model2", "Metric2"] == 5.0
