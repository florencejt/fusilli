import pytest
import torch
from fusilli.eval import ParentPlotter, RealsVsPreds, ConfusionMatrix, ModelComparison
from unittest.mock import Mock
import pandas as pd
import shutil

# check that subspace method from_new_data throws error with incorrect checkpoint file name


# @pytest.fixture(scope="module")
# def create_test_data(tmp_path_factory):
#     # Create a temporary directory
#     tmp_dir = tmp_path_factory.mktemp("test_data")

#     # Create sample CSV files with different index and label column names
#     tabular1_csv = tmp_dir / "tabular1_new_data.csv"
#     tabular1_data = pd.DataFrame(
#         {
#             "study_id": range(10),  # Different index column name
#             "feature1": [1.0] * 10,
#             "feature2": [2.0] * 10,
#             "pred_label": [0] * 10,  # Different label column name
#         }
#     )
#     tabular1_data.to_csv(tabular1_csv, index=False)

#     tabular2_csv = tmp_dir / "tabular2_new_data.csv"
#     tabular2_data = pd.DataFrame(
#         {
#             "study_id": range(10),
#             "feature3": [3.0] * 10,
#             "feature4": [4.0] * 10,
#             "pred_label": [1] * 10,
#         }
#     )
#     tabular2_data.to_csv(tabular2_csv, index=False)

#     # Create a sample Torch file for image data
#     image_data_2d = torch.randn(10, 1, 100, 100)
#     image_torch_file_2d = tmp_dir / "image_data_2d_new_data.pt"
#     torch.save(image_data_2d, image_torch_file_2d)

#     # Create a sample Torch file for image data
#     image_data_3d = torch.randn(10, 1, 100, 100, 100)
#     image_torch_file_3d = tmp_dir / "image_data_3d_new_data.pt"
#     torch.save(image_data_3d, image_torch_file_3d)

#     yield {
#         "tabular1_csv": tabular1_csv,
#         "tabular2_csv": tabular2_csv,
#         "image_torch_file_2d": image_torch_file_2d,
#         "image_torch_file_3d": image_torch_file_3d,
#     }

#     # Clean up temporary files and directories
#     shutil.rmtree(tmp_dir)


# # Define a fixture to create a sample model_list
# @pytest.fixture
# def sample_model_list():
#     # Create a list of mock fold models for testing
#     class MockFoldModel:
#         def __init__(self):
#             self.train_reals = torch.tensor([1.0, 2.0, 3.0])
#             self.train_preds = torch.tensor([1.1, 2.2, 3.3])
#             self.val_reals = torch.tensor([4.0, 5.0, 6.0])
#             self.val_preds = torch.tensor([4.4, 5.5, 6.6])
#             self.val_logits = torch.tensor([0.1, 0.2, 0.3])
#             self.metric1 = 0.9
#             self.metric2 = 0.8

#     return [MockFoldModel(), MockFoldModel()]


# # Test cases for ParentPlotter class methods
# def test_get_kfold_data_from_model(sample_model_list, create_test_data):
#     (
#         train_reals,
#         train_preds,
#         val_reals,
#         val_preds,
#         metrics_per_fold,
#         overall_kfold_metrics,
#     ) = ParentPlotter.get_kfold_data_from_model(sample_model_list)

#     # Perform assertions based on your test data and expectations
#     assert len(train_reals) == 2
#     assert len(train_preds) == 2
#     assert len(val_reals) == 2
#     assert len(val_preds) == 2
#     assert isinstance(metrics_per_fold, dict)
#     assert isinstance(overall_kfold_metrics, dict)


# def test_get_tt_data_from_model():
#     # Create a mock model for testing
#     class MockModel:
#         def __init__(self):
#             self.train_reals = torch.tensor([1.0, 2.0, 3.0])
#             self.train_preds = torch.tensor([1.1, 2.2, 3.3])
#             self.val_reals = torch.tensor([4.0, 5.0, 6.0])
#             self.val_preds = torch.tensor([4.4, 5.5, 6.6])
#             self.metrics = {
#                 "type": [
#                     {"name": "metric1", "metric": lambda x, y: 0.9},
#                     {"name": "metric2", "metric": lambda x, y: 0.8},
#                 ]
#             }

#         def eval(self):
#             pass

#     model = [MockModel()]
#     (
#         train_reals,
#         train_preds,
#         val_reals,
#         val_preds,
#         metric_values,
#     ) = ParentPlotter.get_tt_data_from_model(model)

#     # Perform assertions based on your test data and expectations
#     assert len(train_reals) == 1
#     assert len(train_preds) == 1
#     assert len(val_reals) == 1
#     assert len(val_preds) == 1
#     assert isinstance(metric_values, dict)


# def test_get_new_kfold_data(sample_model_list):
#     params = {}  # Define sample params as needed
#     data_file_suffix = "_new_data"
#     checkpoint_file_suffix = "_firsttry"

#     (
#         train_reals,
#         train_preds,
#         val_reals,
#         val_preds,
#         metrics_per_fold,
#         overall_kfold_metrics,
#     ) = ParentPlotter.get_new_kfold_data(
#         sample_model_list, params, data_file_suffix, checkpoint_file_suffix
#     )

#     # Perform assertions based on your test data and expectations
#     assert len(train_reals) == 2
#     assert len(train_preds) == 2
#     assert len(val_reals) == 2
#     assert len(val_preds) == 2
#     assert isinstance(metrics_per_fold, dict)
#     assert isinstance(overall_kfold_metrics, dict)


# def test_get_new_tt_data(create_test_data):
#     # Create a mock model for testing

#     class MockFusionModel:
#         fusion_type = "attention"
#         modality_type = "both_tab"

#         def __init__(self, pred_type, data_dims, params):
#             pass

#     class MockModel:
#         def __init__(self):
#             self.train_reals = torch.tensor([1.0, 2.0, 3.0])
#             self.train_preds = torch.tensor([1.1, 2.2, 3.3])
#             self.val_reals = torch.tensor([4.0, 5.0, 6.0])
#             self.val_preds = torch.tensor([4.4, 5.5, 6.6])
#             self.metrics = {
#                 "type": [
#                     {"name": "metric1", "metric": lambda x, y: 0.9},
#                     {"name": "metric2", "metric": lambda x, y: 0.8},
#                 ]
#             }
#             # self.model = Mock(fusion_type="attention")
#             self.model = MockFusionModel("regression", 2, None)
#             self.model.subspace_method = None
#             # self.model.fusion_type = "attention"
#             # del self.model.graph_maker  # Remove graph_maker attribute to avoid errors

#         def eval(self):
#             pass

#     # create tmpdir for checkpoint_dir
#     tmp_dir = create_test_data["tabular1_csv"].parent / "tmp_dir"
#     tmp_dir.mkdir()

#     # put in fake checkpoint file called MockFusionModel_firsttry.ckpt
#     checkpoint_file = tmp_dir / "MockFusionModel_firsttry.ckpt"
#     checkpoint_file.touch()

#     model = [MockModel()]
#     params = {
#         "kfold_flag": False,
#         "test_size": 0.3,
#         "pred_type": "regression",
#         "checkpoint_dir": tmp_dir,
#     }  # Define sample params as needed
#     data_file_suffix = "_new_data"
#     checkpoint_file_suffix = "_firsttry"

#     tabular1_csv = create_test_data["tabular1_csv"]
#     tabular2_csv = create_test_data["tabular2_csv"]
#     image_torch_file_2d = create_test_data["image_torch_file_2d"]

#     params["tabular1_source_new_data"] = tabular1_csv
#     params["tabular2_source_new_data"] = tabular2_csv
#     params["img_source_new_data"] = image_torch_file_2d

#     (
#         train_reals,
#         train_preds,
#         val_reals,
#         val_preds,
#         metric_values,
#     ) = ParentPlotter.get_new_tt_data(
#         model, params, data_file_suffix, checkpoint_file_suffix
#     )

#     # Perform assertions based on your test data and expectations
#     assert len(train_reals) == 1
#     assert len(train_preds) == 1
#     assert len(val_reals) == 1
#     assert len(val_preds) == 1
#     assert isinstance(metric_values, dict)


# # You can add more test cases as needed
