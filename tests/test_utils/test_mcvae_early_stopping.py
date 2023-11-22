# from fusilli.fusionmodels.tabularfusion.mcvae_model import mcvae_early_stopping_tol
# import pytest
#
#
# def test_mcvae_early_stopping_tol_no_early_stop():
#     # Test when there's no early stopping (i.e., patience and tolerance are not met)
#     patience = 5
#     tolerance = 1e-4
#     loss_logs = [0.1, 0.2, 0.3, 0.4, 0.5]
#     result = mcvae_early_stopping_tol(patience, tolerance, loss_logs)
#     assert result == len(loss_logs) - 1  # Should return the last epoch
#
#
# def test_mcvae_early_stopping_tol_with_early_stop():
#     # Test when early stopping conditions are met
#     patience = 3
#     tolerance = 1e-4
#     loss_logs = [0.1, 0.2, 0.3, 0.3, 0.3, 0.3, 0.4, 0.5]
#     result = mcvae_early_stopping_tol(patience, tolerance, loss_logs)
#     assert result == 2  # Should return the epoch where conditions are met (epoch 2)
