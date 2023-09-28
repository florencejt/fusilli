import pytest

from fusilli.eval import Plotter
from fusilli.fusion_models.base_model import BaseModel
import torch.nn as nn


class model_instance1(nn.Module):
    def __init__(self, params):
        super(model_instance1, self).__init__()

        self.pred_type = params["pred_type"]


def test_plotter_init():
    # Test the initialization of the Plotter class
    params = {"pred_type": "regression", "kfold_flag": False}
    trained_model_dict = {"model1": BaseModel(model_instance1(params))}

    plotter = Plotter(trained_model_dict, params)

    # Check that the attributes are correctly initialized
    assert plotter.trained_model_dict == trained_model_dict
    assert plotter.model_names == list(trained_model_dict.keys())
    assert plotter.params == params
    assert plotter.pred_type == params["pred_type"]


def test_get_kfold_numbers():
    params = {"pred_type": "regression", "kfold_flag": True, "num_k": 5}

    trained_model_dict = {
        "model1": [BaseModel(model_instance1(params))] * params["num_k"]
    }
    plotter = Plotter(trained_model_dict, params)

    # Call the method you want to test
    plotter.plot_all()

    # # Add assertions to check the behavior of the method
    # assert len(plotter.train_reals) == params["num_k"]
    # assert len(plotter.train_preds) == params["num_k"]
    # assert len(plotter.val_reals) == params["num_k"]
    # assert len(plotter.val_preds) == params["num_k"]
    # assert len(plotter.val_logits) == params["num_k"]
