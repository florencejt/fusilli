import copy

import pytest
import torch
import pandas as pd
from unittest.mock import patch, Mock
from unittest import mock
import lightning.pytorch as pl
# from lightning.pytorch.callbacks import EarlyStopping
from torch import nn
from torch_geometric.data import Data

from fusilli.data import CustomDataset

# from fusilli.fusionmodels.tabularfusion.mcvae_model import MCVAESubspaceMethod
from fusilli.fusionmodels.tabularimagefusion.denoise_tab_img_maps import (
    DenoisingAutoencoder,
    ImgUnimodalDAE,
    denoising_autoencoder_subspace_method
)
from fusilli.fusionmodels.tabularfusion.edge_corr_gnn import EdgeCorrGraphMaker

from ..test_data.test_TrainTestDataModule import create_test_files
from fusilli.data import TrainTestDataModule
# from fusilli.utils.mcvae.src.mcvae.models import Mcvae
# from fusilli.utils.training_utils import init_trainer
from fusilli.fusionmodels.tabularimagefusion.concat_img_latent_tab_doubletrain import (
    concat_img_latent_tab_subspace_method, ImgLatentSpace
)

from fusilli.fusionmodels.tabularfusion.attention_weighted_GNN import AttentionWeightedGraphMaker


class MockFusionModel:
    fusion_type = "subspace"
    modality_type = "tabular_tabular"

    def __init__(self):
        pass


@pytest.fixture
def sample_datamodule(create_test_files):
    # Define a sample datamodule object for testing
    # You may need to create a fixture with appropriate data for your tests
    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    image_torch_file_2d = create_test_files["image_torch_file_2d"]

    # using a MockFusionModel with no subspace method so we can test the
    # MCVAESubspaceMethod class in isolation
    fusion_model = MockFusionModel()

    # Call the prepare_fusion_data function with custom fusion type (non-graph)
    dm = TrainTestDataModule(fusion_model=MockFusionModel,
                             sources=[tabular1_csv, tabular2_csv, image_torch_file_2d],
                             output_paths=None,
                             prediction_task="binary",
                             batch_size=8,
                             test_size=0.3,
                             multiclass_dimensions=None,
                             )

    # dm = prepare_fusion_data(fusion_model, params)
    dm.prepare_data()
    dm.setup()

    return dm


class MockFusionTabImgModel:
    fusion_type = "subspace"
    modality_type = "tabular_image"

    def __init__(self):
        pass


@pytest.fixture
def sample_tabimg_datamodule(create_test_files):
    # Define a sample datamodule object for testing
    # You may need to create a fixture with appropriate data for your tests
    tabular1_csv = create_test_files["tabular1_csv"]
    tabular2_csv = create_test_files["tabular2_csv"]
    image_torch_file_2d = create_test_files["image_torch_file_2d"]

    # using a MockFusionModel with no subspace method so we can test the
    # MCVAESubspaceMethod class in isolation
    fusion_model = MockFusionTabImgModel()

    # Call the prepare_fusion_data function with custom fusion type (non-graph)
    dm = TrainTestDataModule(fusion_model=MockFusionTabImgModel,
                             sources=[tabular1_csv, tabular2_csv, image_torch_file_2d],
                             output_paths=None,
                             prediction_task="binary",
                             batch_size=8,
                             test_size=0.3,
                             multiclass_dimensions=None,
                             )

    # dm = prepare_fusion_data(fusion_model, params)
    dm.prepare_data()
    dm.setup()

    return dm


# def test_mcvaesubspacemethod_initialisation(sample_datamodule):
#     # mock the fusilli.utils.training_utils.get_checkpoint_filenames_for_subspace_models
#     # function to return a list of checkpoint filenames
#
#     dm = sample_datamodule
#
#     assert hasattr(MCVAESubspaceMethod, "subspace_models")
#
#     mcvae_subspace = MCVAESubspaceMethod(dm)
#
#     assert mcvae_subspace.datamodule == dm
#     assert mcvae_subspace.num_latent_dims == 10
#
#
# def test_mcvaesubspacemethod_check_params(sample_datamodule):
#     # Test the check_params method
#     mcvae_subspace = MCVAESubspaceMethod(sample_datamodule)
#     mcvae_subspace.check_params()  # Ensure it doesn't raise exceptions


# def test_mcvaesubspacemethod_train(sample_datamodule):
#     # Test the train method
#
#     dm = sample_datamodule
#     train_dataset = dm.train_dataset
#     # train_dataset = sample_train_dataset
#
#     mcvae_subspace = MCVAESubspaceMethod(dm, max_epochs=50)
#     mean_latents, labels = mcvae_subspace.train(train_dataset)
#     assert isinstance(mean_latents, torch.Tensor)
#     assert isinstance(labels, pd.DataFrame)
#
#     # check that mcvae_early_stopping_tol was called only once
#     mock_mcvae_early_stopping_tol = Mock(return_value=35)
#     with patch("fusilli.fusionmodels.tabularfusion.mcvae_model.mcvae_early_stopping_tol",
#                mock_mcvae_early_stopping_tol):
#         mcvae_subspace = MCVAESubspaceMethod(dm, max_epochs=50)
#         mcvae_subspace.train(train_dataset)
#         mock_mcvae_early_stopping_tol.assert_called_once()
#
#     # look at the return values of the train method and ensure they are correct
#     # (i.e. the correct number of latent dimensions are returned)
#     mcvae_subspace = MCVAESubspaceMethod(dm, max_epochs=50)
#     mean_latents, labels = mcvae_subspace.train(train_dataset)
#     assert mean_latents.shape[1] == mcvae_subspace.num_latent_dims
#     assert labels.shape[1] == 1
#
#
# def test_mcvaesubspacemethod_convert_to_latent(sample_datamodule):
#     # Test the convert_to_latent method
#
#     dm = sample_datamodule
#     train_dataset = dm.train_dataset
#     test_dataset = dm.test_dataset
#
#     mcvae_subspace = MCVAESubspaceMethod(dm, max_epochs=50)
#
#     # raise error if train() has not been called - means that the model doesn't have the attribute 'fit_model' yet
#     with pytest.raises(AttributeError, match=r"fit_model"):
#         mcvae_subspace.convert_to_latent(test_dataset)
#
#     # train the model
#     mcvae_subspace.train(train_dataset)
#
#     # check that convert_to_latent() returns the correct values
#     test_mean_latents, labels, dimensions = mcvae_subspace.convert_to_latent(test_dataset)
#
#     assert isinstance(test_mean_latents, torch.Tensor)
#     assert test_mean_latents.shape[1] == mcvae_subspace.num_latent_dims
#     assert isinstance(labels, pd.DataFrame)
#     assert len(labels) == len(test_mean_latents)
#     assert isinstance(dimensions, list)
#     assert dimensions[1] == None
#     assert dimensions[2] == None
#     assert len(dimensions) == 3


# DENOISING AUTOENCODER MODEL
def test_denoising_autoencoder_initialisation():
    # Test the initialization of the DenoisingAutoencoder
    data_dims = [128]  # Replace with actual data dimensions
    model = DenoisingAutoencoder(data_dims)

    assert isinstance(model, pl.LightningModule)
    assert isinstance(model.upsampler, nn.Sequential)
    assert isinstance(model.downsampler, nn.Sequential)
    assert isinstance(model.loss, nn.MSELoss)
    assert hasattr(model, "latent_dim")


def test_denoising_autoencoder_forward():
    # Test the forward method of the DenoisingAutoencoder
    data_dims = [128]  # Replace with actual data dimensions
    model = DenoisingAutoencoder(data_dims)
    x = torch.randn(1, data_dims[0])  # Replace with actual input data
    output, x_before_dropout = model(x)

    assert output.shape == x.shape
    assert torch.eq(x, x_before_dropout).all()


@pytest.mark.filterwarnings("ignore:.*You are trying to `self.log()`*.", )
def test_denoising_autoencoder_training_step():
    # Test the training_step method of the DenoisingAutoencoder
    data_dims = [128]  # Replace with actual data dimensions
    model = DenoisingAutoencoder(data_dims)
    x = torch.randn(1, data_dims[0])  # Replace with actual input data
    loss = model.training_step(x, 0)

    assert isinstance(loss, torch.Tensor)
    assert len(loss.shape) == 0


@pytest.mark.filterwarnings("ignore:.*You are trying to `self.log()`*.", )
def test_denoising_autoencoder_validation_step():
    # Test the validation_step method of the DenoisingAutoencoder
    data_dims = [128]  # Replace with actual data dimensions
    model = DenoisingAutoencoder(data_dims)
    x = torch.randn(1, data_dims[0])  # Replace with actual input data
    loss = model.validation_step(x, 0)

    assert isinstance(loss, torch.Tensor)
    assert len(loss.shape) == 0


# IMG LATENT SPACE MODEL
def test_img_unimodal_dae_initialisation():
    # Test the initialization of the ImgUnimodalDAE
    data_dims = [None, None, (100, 100)]
    pred_type = "multiclass"
    multiclass_dims = 3
    model = ImgUnimodalDAE(data_dims, pred_type, multiclass_dims)

    assert isinstance(model, pl.LightningModule)
    assert isinstance(model.img_layers, nn.ModuleDict)
    assert hasattr(model, "loss")
    assert hasattr(model, "activation")
    assert hasattr(model, "prediction_task")
    assert model.prediction_task == pred_type
    assert hasattr(model, "multiclass_dimensions")
    assert model.multiclass_dimensions == multiclass_dims


def test_img_unimodal_dae_forward():
    # Test the forward method of the ImgUnimodalDAE
    data_dims = [None, None, (100, 100)]
    pred_type = "multiclass"
    multiclass_dims = 3
    model = ImgUnimodalDAE(data_dims, pred_type, multiclass_dims)
    image_data = torch.rand((1, 1, data_dims[2][0], data_dims[2][1]))
    output = model(image_data)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, multiclass_dims)


@pytest.mark.filterwarnings("ignore:.*You are trying to `self.log()`*.", )
def test_img_unimodal_dae_training_step():
    # Test the training_step method of the ImgUnimodalDAE
    data_dims = [None, None, (100, 100)]
    pred_type = "multiclass"
    multiclass_dims = 3
    model = ImgUnimodalDAE(data_dims, pred_type, multiclass_dims)
    image_data = torch.rand((3, 1, data_dims[2][0], data_dims[2][1]))
    labels = torch.randint(0, multiclass_dims, (3,))
    batch = (torch.randn(1, 10), image_data, labels)
    loss = model.training_step(batch, 0)

    assert isinstance(loss, torch.Tensor)
    assert len(loss.shape) == 0


@pytest.mark.filterwarnings("ignore:.*You are trying to `self.log()`*.", )
def test_img_unimodal_dae_validation_step():
    # Test the validation_step method of the ImgUnimodalDAE
    data_dims = [None, None, (100, 100)]
    pred_type = "multiclass"
    multiclass_dims = 3
    model = ImgUnimodalDAE(data_dims, pred_type, multiclass_dims)
    image_data = torch.rand((3, 1, data_dims[2][0], data_dims[2][1]))
    labels = torch.randint(0, multiclass_dims, (3,))
    batch = (torch.randn(1, 10), image_data, labels)
    loss = model.validation_step(batch, 0)

    assert isinstance(loss, torch.Tensor)
    assert len(loss.shape) == 0


def test_img_unimodal_dae_get_intermediate_maps():
    # Test the get_intermediate_maps method of the ImgUnimodalDAE
    data_dims = [None, None, (100, 100)]
    pred_type = "multiclass"
    multiclass_dims = 3
    model = ImgUnimodalDAE(data_dims, pred_type, multiclass_dims)
    image_data = torch.rand((3, 1, data_dims[2][0], data_dims[2][1]))
    labels = torch.randint(0, multiclass_dims, (3,))
    batch = (torch.randn(1, 10), image_data, labels)
    output = model.get_intermediate_featuremaps(image_data)

    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == 3  # batch size
    assert len(output.shape) == 2  # flattened


@pytest.fixture(name="mock_init_trainer", scope="function")
def fixture_mock_init():
    def _mock_init_trainer(*args, **kwargs):
        class trainer:
            def __init__(self):
                pass

            def fit(self, *args, **kwargs):
                return torch.randn(10, 10), pd.DataFrame([1, 2, 3, 4, 5])

            def validate(self, *args, **kwargs):
                return torch.randn(10, 10), pd.DataFrame([1, 2, 3, 4, 5])

        return trainer()

    return _mock_init_trainer


def test_denoising_autoencoder_subspace_method_init(sample_tabimg_datamodule, mock_init_trainer):
    # mock_init_trainer = mocker.patch("fusilli.utils.training_utils.init_trainer")
    # mock_init_trainer.return_value = 10
    mock_init_trainer = mock.create_autospec(mock_init_trainer)
    mock_patch = ("fusilli.fusionmodels.tabularimagefusion.denoise_tab_img_maps.init_trainer")
    with patch(mock_patch) as mck:
        mck.side_effect = mock_init_trainer

        assert hasattr(denoising_autoencoder_subspace_method, "subspace_models")
        assert len(denoising_autoencoder_subspace_method.subspace_models) == 2
        assert denoising_autoencoder_subspace_method.subspace_models[0] == DenoisingAutoencoder
        assert denoising_autoencoder_subspace_method.subspace_models[1] == ImgUnimodalDAE

        dm = sample_tabimg_datamodule

        dae_subspace_method = denoising_autoencoder_subspace_method(dm)

        # assert init trainer called twice for the two subspace methods
        assert mock_init_trainer.call_count == 2

        assert hasattr(dae_subspace_method, "autoencoder")
        assert isinstance(dae_subspace_method.autoencoder, DenoisingAutoencoder)
        assert hasattr(dae_subspace_method, "img_unimodal")
        assert isinstance(dae_subspace_method.img_unimodal, ImgUnimodalDAE)


@pytest.fixture(name="mock_load_from_checkpoint", scope="function")
def fixture_mock_load_chkpt():
    def _mock_load_from_checkpoint(*args, **kwargs):
        return 10

    return _mock_load_from_checkpoint


@pytest.fixture(name="mock_torch_load", scope="function")
def fixture_mock_torch_load():
    def _mock_torch_load(*args, **kwargs):
        return {"state_dict": 30}

    return _mock_torch_load


def test_denoising_autoencoder_subspace_method_init_with_checkpoint(sample_tabimg_datamodule,
                                                                    mock_load_from_checkpoint, mock_init_trainer,
                                                                    mock_torch_load):
    mock_load_from_checkpoint = mock.create_autospec(mock_load_from_checkpoint)
    mock_init_trainer = mock.create_autospec(mock_init_trainer)
    mock_patch = ("fusilli.fusionmodels.tabularimagefusion.denoise_tab_img_maps.DenoisingAutoencoder."
                  "load_state_dict")
    mock_patch12 = ("fusilli.fusionmodels.tabularimagefusion.denoise_tab_img_maps.ImgUnimodalDAE."
                    "load_state_dict")
    mock_patch2 = ("fusilli.fusionmodels.tabularimagefusion.denoise_tab_img_maps.init_trainer")
    mock_patch3 = ("fusilli.fusionmodels.tabularimagefusion.denoise_tab_img_maps.torch.load")
    with patch(mock_patch) as mck, patch(mock_patch2) as mck2, patch(mock_patch12) as mck12, patch(mock_patch3) as mck3:
        mck.side_effect = mock_load_from_checkpoint
        mck12.side_effect = mock_load_from_checkpoint
        mck2.side_effect = mock_init_trainer
        mck3.side_effect = mock_torch_load

        dm = sample_tabimg_datamodule

        dae_subspace_method = denoising_autoencoder_subspace_method(dm, train_subspace=False)

        dae_subspace_method.load_ckpt(["path1", "path2"])

        # assert init trainer called twice for the two subspace methods
        assert mock_init_trainer.call_count == 0
        # assert load_from_checkpoint called twice for the two subspace methods
        assert mock_load_from_checkpoint.call_count == 2


class Subspace:
    def __init__(self, *args, **kwargs):
        pass

    def denoise(self, *args, **kwargs):
        return torch.randn(10, 10)

    def get_intermediate_featuremaps(self, *args, **kwargs):
        return torch.randn(10, 10)

    def eval(self):
        pass


# def test_denoising_autoencoder_subspace_method_train(sample_tabimg_datamodule, mock_init_trainer):
#     dm = sample_tabimg_datamodule
#     train_dataset = dm.train_dataset
#     test_dataset = dm.test_dataset
#
#     mock_patch = ("fusilli.fusionmodels.tabularimagefusion.denoise_tab_img_maps.init_trainer")
#
#     class trainer:
#         def __init__(self):
#             pass
#
#         def fit(self, *args, **kwargs):
#             return torch.randn(10, 7), pd.DataFrame([0] * 10, columns=["prediction_label"])
#
#         def validate(self, *args, **kwargs):
#             return torch.randn(10, 3), pd.DataFrame([0] * 10, columns=["prediction_label"])
#
#     with patch(mock_patch) as mck:
#         mck.side_effect = mock_init_trainer
#
#         # copy a version of the denoising_autoencoder_subspace_method
#         copy_dae_method = copy.deepcopy(denoising_autoencoder_subspace_method)
#         copy_dae_method.subspace_models = [Subspace, Subspace]
#         dae_subspace_method = copy_dae_method(dm, checkpoint_path=None)
#
#         mean_latents, labels = dae_subspace_method.train(train_dataset, test_dataset)
#         mean_latents_val, labels_val, data_dims = dae_subspace_method.convert_to_latent(test_dataset)
#

# GRAPH MAKER

data1 = torch.randn(100, 15)
data2 = torch.randn(100, 25)
labels = pd.DataFrame({"prediction_label": [0] * 100})
dummy_dataset = CustomDataset([data1, data2], labels)


def test_edge_corr_graph_maker():
    # Create an instance of EdgeCorrGraphMaker
    edge_corr_graph_maker = EdgeCorrGraphMaker(dummy_dataset)

    # Check the default threshold
    assert edge_corr_graph_maker.threshold == 0.8

    # Check if parameters are valid
    with pytest.raises(ValueError):
        edge_corr_graph_maker.threshold = -0.1
        edge_corr_graph_maker.check_params()

    edge_corr_graph_maker.threshold = 0.5

    # Create a graph
    graph_data = edge_corr_graph_maker.make_graph()

    # Assertions
    assert isinstance(graph_data, Data)
    assert graph_data.x.shape == dummy_dataset[:][1].shape
    assert graph_data.edge_index.shape[1] > 0
    assert graph_data.edge_attr.shape[0] == graph_data.edge_index.shape[1]

    # assert that all the edge weights are above the threshold
    assert (abs(graph_data.edge_attr - 1) >= edge_corr_graph_maker.threshold).all()


@pytest.mark.filterwarnings("ignore:.*does not have many workers which may be a bottleneck*.", )
def test_AttentionWeightedGraphMaker():
    # Create an instance of EdgeCorrGraphMaker
    attention_weighted_graph_maker = AttentionWeightedGraphMaker(dummy_dataset)

    # Check the default threshold
    assert attention_weighted_graph_maker.edge_probability_threshold == 75
    assert attention_weighted_graph_maker.attention_MLP_test_size == 0.2
    assert hasattr(attention_weighted_graph_maker, "early_stop_callback")

    # Check if parameters are valid
    with pytest.raises(TypeError):
        attention_weighted_graph_maker.edge_probability_threshold = -0.1
        attention_weighted_graph_maker.check_params()

    with pytest.raises(ValueError):
        attention_weighted_graph_maker.edge_probability_threshold = 110
        attention_weighted_graph_maker.check_params()

    with pytest.raises(ValueError):
        attention_weighted_graph_maker.attention_MLP_test_size = -0.1
        attention_weighted_graph_maker.check_params()

    with pytest.raises(ValueError):
        attention_weighted_graph_maker.attention_MLP_test_size = 1.1
        attention_weighted_graph_maker.check_params()

    with pytest.raises(ValueError):
        attention_weighted_graph_maker.early_stop_callback = 1
        attention_weighted_graph_maker.check_params()

    new_instance = AttentionWeightedGraphMaker(dummy_dataset)

    new_instance.edge_probability_threshold = 85
    new_instance.attention_MLP_test_size = 0.4

    # Create a graph
    graph_data = new_instance.make_graph()

    # Assertions
    assert isinstance(graph_data, Data)
    assert graph_data.x.shape == dummy_dataset[:][1].shape
    assert graph_data.edge_index.shape[1] > 0
    assert graph_data.edge_attr.shape[0] == graph_data.edge_index.shape[1]


# concat img and tabular data latent space double train
def test_ImgLatentSpace_init():
    # Test the initialization of the ImgLatentSpace
    data_dims = [None, None, (100, 100)]
    model = ImgLatentSpace(data_dims)

    assert isinstance(model, pl.LightningModule)
    assert isinstance(model.encoder, nn.Sequential)
    assert isinstance(model.decoder, nn.Sequential)
    assert hasattr(model, "latent_dim")
    assert model.latent_dim == 64


def test_ImgLatentSpace_forward():
    # Test the forward method of the ImgLatentSpace
    data_dims = [None, None, (100, 100)]
    model = ImgLatentSpace(data_dims)
    image_data = torch.rand((1, 1, data_dims[2][0], data_dims[2][1]))
    output = model(image_data)

    assert isinstance(output, torch.Tensor)
    assert output.shape == image_data.shape


@pytest.mark.filterwarnings("ignore:.*You are trying to `self.log()`*.", )
def test_ImgLatentSpace_training_step():
    # Test the training_step method of the ImgLatentSpace
    data_dims = [None, None, (100, 100)]
    model = ImgLatentSpace(data_dims)
    image_data = torch.rand((5, 1, data_dims[2][0], data_dims[2][1]))
    loss = model.training_step(image_data, 0)

    assert isinstance(loss, torch.Tensor)
    assert len(loss.shape) == 0


@pytest.mark.filterwarnings("ignore:.*You are trying to `self.log()`*.", )
def test_ImgLatentSpace_validation_step():
    # Test the training_step method of the ImgLatentSpace
    data_dims = [None, None, (100, 100)]
    model = ImgLatentSpace(data_dims)
    image_data = torch.rand((5, 1, data_dims[2][0], data_dims[2][1]))
    loss = model.validation_step(image_data, 0)

    assert isinstance(loss, torch.Tensor)
    assert len(loss.shape) == 0


def test_ImgLatentSpace_encode_image():
    # Test the encode_image method of the ImgLatentSpace
    data_dims = [None, None, (100, 100)]
    model = ImgLatentSpace(data_dims)
    image_data = torch.rand((5, 1, data_dims[2][0], data_dims[2][1]))
    output = model.encode_image(image_data)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (5, 64)


def test_concat_img_latent_tab_subspace_method_init(sample_tabimg_datamodule, mock_init_trainer):
    # mock_init_trainer = mocker.patch("fusilli.utils.training_utils.init_trainer")
    # mock_init_trainer.return_value = 10
    mock_init_trainer = mock.create_autospec(mock_init_trainer)
    mock_patch = ("fusilli.fusionmodels.tabularimagefusion.concat_img_latent_tab_doubletrain.init_trainer")
    with patch(mock_patch) as mck:
        mck.side_effect = mock_init_trainer

        assert hasattr(concat_img_latent_tab_subspace_method, "subspace_models")
        assert len(concat_img_latent_tab_subspace_method.subspace_models) == 1
        assert concat_img_latent_tab_subspace_method.subspace_models[0] == ImgLatentSpace

        dm = sample_tabimg_datamodule

        img_latent_subspace_method = concat_img_latent_tab_subspace_method(dm)

        # assert init trainer called twice for the two subspace methods
        assert mock_init_trainer.call_count == 1

        assert hasattr(img_latent_subspace_method, "autoencoder")
        assert isinstance(img_latent_subspace_method.autoencoder, ImgLatentSpace)


def test_concat_img_latent_tab_subspace_method_init_with_checkpoint(sample_tabimg_datamodule,
                                                                    mock_load_from_checkpoint, mock_init_trainer,
                                                                    mock_torch_load):
    mock_load_from_checkpoint = mock.create_autospec(mock_load_from_checkpoint)
    mock_init_trainer = mock.create_autospec(mock_init_trainer)
    mock_patch = ("fusilli.fusionmodels.tabularimagefusion.concat_img_latent_tab_doubletrain.ImgLatentSpace."
                  "load_state_dict")
    mock_patch2 = ("fusilli.fusionmodels.tabularimagefusion.concat_img_latent_tab_doubletrain.init_trainer")
    mock_patch3 = ("fusilli.fusionmodels.tabularimagefusion.concat_img_latent_tab_doubletrain.torch.load")
    with patch(mock_patch) as mck, patch(mock_patch2) as mck2, patch(mock_patch3) as mck3:
        mck.side_effect = mock_load_from_checkpoint
        mck2.side_effect = mock_init_trainer
        mck3.side_effect = mock_torch_load

        dm = sample_tabimg_datamodule

        img_latent_subspace = concat_img_latent_tab_subspace_method(dm, train_subspace=False)

        img_latent_subspace.load_ckpt(["path1"])

        # assert init trainer called twice for the two subspace methods
        assert mock_init_trainer.call_count == 0
        # assert load_from_checkpoint called twice for the two subspace methods
        assert mock_load_from_checkpoint.call_count == 1


def test_concat_img_latent_tab_subspace_method_train(sample_tabimg_datamodule, mock_init_trainer):
    dm = sample_tabimg_datamodule
    train_dataset = dm.train_dataset
    test_dataset = dm.test_dataset

    mock_patch = ("fusilli.fusionmodels.tabularimagefusion.concat_img_latent_tab_doubletrain.init_trainer")

    class trainer:
        def __init__(self):
            pass

        def fit(self, *args, **kwargs):
            return torch.randn(10, 7), pd.DataFrame([0] * 10, columns=["prediction_label"])

        def validate(self, *args, **kwargs):
            return torch.randn(10, 3), pd.DataFrame([0] * 10, columns=["prediction_label"])

    with patch(mock_patch) as mck:
        mck.side_effect = mock_init_trainer

        concat_img_latent_tab_subspace_method.subspace_models = [Subspace]
        img_latent_subspace_method = concat_img_latent_tab_subspace_method(dm)

        mean_latents, labels = img_latent_subspace_method.train(train_dataset, test_dataset)
        mean_latents_val, labels_val, data_dims = img_latent_subspace_method.convert_to_latent(test_dataset)
