import pytest

from fusionlibrary.eval import Plotter


def test_plotter_init():
    # Test the initialization of the Plotter class
    trained_model_dict = {"model1": "model_instance1", "model2": "model_instance2"}
    params = {"pred_type": "regression", "kfold_flag": False}

    plotter = Plotter(trained_model_dict, params)

    # Check that the attributes are correctly initialized
    assert plotter.trained_model_dict == trained_model_dict
    assert plotter.model_names == list(trained_model_dict.keys())
    assert plotter.params == params
    assert plotter.pred_type == params["pred_type"]
