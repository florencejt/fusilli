# import pytest
# import torch
# import torch.nn as nn
# import torchmetrics as tm
# import torch.nn.functional as F
# from fusionlibrary.fusion_models.base_model import BaseModel


# # Sample Fusion Model for Testing
# class SampleFusionModel(nn.Module):
#     def __init__(self):
#         super(SampleFusionModel, self).__init__()
#         self.pred_type = "binary"  # Set prediction type for testing
#         self.multiclass_dim = 1  # Set multiclass dimension for testing
#         self.fusion_type = "attention"  # Set fusion type for testing

#     def forward(self, x):
#         # Sample forward function
#         return [
#             torch.randn(1),
#         ]


# # Sample Data for Testing
# class SampleBatch:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y


# # Define test cases
# class TestBaseModel:
#     @pytest.fixture
#     def base_model(self):
#         fusion_model = SampleFusionModel()
#         return BaseModel(fusion_model)

#     def test_safe_squeeze_1d_tensor(self, base_model):
#         tensor = torch.tensor([1.0])
#         squeezed = base_model.safe_squeeze(tensor)
#         assert squeezed.shape == (1,)

#     def test_safe_squeeze_2d_tensor(self, base_model):
#         tensor = torch.tensor([[1.0]])
#         squeezed = base_model.safe_squeeze(tensor)
#         assert squeezed.shape == (1,)

#     def test_get_data_from_batch(self, base_model):
#         x = torch.randn(1, 2)
#         y = torch.randn(1)
#         batch = SampleBatch(x, y)
#         data_x, data_y = base_model.get_data_from_batch(batch)
#         assert torch.equal(data_x, x)
#         assert torch.equal(data_y, y)

#     def test_get_model_outputs(self, base_model):
#         x = torch.randn(1, 2)
#         output = base_model.get_model_outputs(x)
#         assert isinstance(output, list)
#         assert len(output) == 1
#         assert isinstance(output[0], torch.Tensor)

#     def test_get_model_outputs_and_loss(self, base_model):
#         x = torch.randn(1, 2)
#         y = torch.randn(1)
#         loss, end_output, logits = base_model.get_model_outputs_and_loss(x, y)
#         assert isinstance(loss, torch.Tensor)
#         assert isinstance(end_output, torch.Tensor)
#         assert isinstance(logits, torch.Tensor)

#     def test_training_step(self, base_model):
#         batch = SampleBatch(torch.randn(1, 2), torch.randn(1))
#         loss = base_model.training_step(batch, batch_idx=0)
#         assert isinstance(loss, torch.Tensor)
#         # Additional assertions can be added

#     def test_validation_step(self, base_model):
#         batch = SampleBatch(torch.randn(1, 2), torch.randn(1))
#         base_model.validation_step(batch, batch_idx=0)
#         # No assertion as it performs logging

#     def test_validation_epoch_end(self, base_model):
#         # Initialize some data
#         batch_val_reals = [torch.randn(1) for _ in range(3)]
#         batch_val_preds = [torch.randn(1) for _ in range(3)]
#         base_model.batch_val_reals = batch_val_reals
#         base_model.batch_val_preds = batch_val_preds

#         # Call validation_epoch_end
#         base_model.validation_epoch_end([])

#         # Assert that batch data is cleared
#         assert base_model.batch_val_reals == []
#         assert base_model.batch_val_preds == []

#         # Additional assertions can be added

#     def test_configure_optimizers(self, base_model):
#         optimizer = base_model.configure_optimizers()
#         assert isinstance(optimizer, torch.optim.Optimizer)


# # Run pytest
# if __name__ == "__main__":
#     pytest.main()
