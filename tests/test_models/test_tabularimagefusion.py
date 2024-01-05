import pytest
import torch
import torch.nn as nn

from fusilli.utils.model_chooser import import_chosen_fusion_models
from fusilli.fusionmodels.base_model import ParentFusionModel

fusion_models = import_chosen_fusion_models({
    "modality_type": ["tabular_image"],
}, skip_models=["MCVAE_tab"])

fusion_model_names = [model.__name__ for model in fusion_models]
# make into dict
fusion_model_dict = {fusion_model_names[i]: fusion_models[i] for i in range(len(fusion_model_names))}
print(fusion_model_dict)


def test_ConcatImageMapsTabularData():
    test_model = fusion_model_dict["ConcatImageMapsTabularData"]

    # attributes before initialising
    assert hasattr(test_model, "method_name")
    assert test_model.method_name == "Concatenating tabular data with image feature maps"
    assert hasattr(test_model, "modality_type")
    assert test_model.modality_type == "tabular_image"
    assert hasattr(test_model, "fusion_type")
    assert test_model.fusion_type == "operation"

    # attributes after initialising
    # modality 1 with 15 features, no modality 2, image with 100x100 2D
    test_model = test_model("binary", [15, None, (100, 100)], None)

    assert hasattr(test_model, "prediction_task")
    assert test_model.prediction_task == "binary"
    assert hasattr(test_model, "img_layers")
    assert hasattr(test_model, "fused_layers")
    assert hasattr(test_model, "final_prediction")
    assert hasattr(test_model, "fused_dim")
    assert test_model.fused_dim > 15  # 15 from tabular data, rest from image after layers and flattening

    # test forward pass
    tabular_data = torch.rand((1, 15))
    image_data = torch.rand((1, 1, 100, 100))
    # forward pass
    test_output = test_model.forward((tabular_data, image_data))
    # check output is correct size
    assert isinstance(test_output, list)
    assert len(test_output) == 1
    assert isinstance(test_output[0], torch.Tensor)
    assert test_output[0].shape == (1, 1)

    # wrong number of modalities
    with pytest.raises(ValueError, match=r"Wrong number of inputs for model!"):
        test_model.forward(tuple(tabular_data))
    with pytest.raises(ValueError, match=r"Wrong number of inputs for model!"):
        test_model.forward((tabular_data, tabular_data, image_data))
    with pytest.raises(TypeError, match=r"Wrong input type for model! Expected tuple"):
        test_model.forward(tabular_data)


def test_ConcatImageMapsTabularMaps():
    test_model = fusion_model_dict["ConcatImageMapsTabularMaps"]

    # attributes before initialising
    assert hasattr(test_model, "method_name")
    assert test_model.method_name == "Concatenating tabular and image feature maps"
    assert hasattr(test_model, "modality_type")
    assert test_model.modality_type == "tabular_image"
    assert hasattr(test_model, "fusion_type")
    assert test_model.fusion_type == "operation"

    test_model = test_model("binary", [15, None, (100, 100)], None)

    assert hasattr(test_model, "prediction_task")
    assert test_model.prediction_task == "binary"
    assert hasattr(test_model, "img_layers")
    assert hasattr(test_model, "mod1_layers")
    assert hasattr(test_model, "fused_layers")
    assert hasattr(test_model, "final_prediction")
    assert hasattr(test_model, "fused_dim")
    assert test_model.fused_dim > 15  # 15 from tabular data, rest from image after layers and flattening

    # test forward pass
    tabular_data = torch.rand((1, 15))
    image_data = torch.rand((1, 1, 100, 100))
    # forward pass
    test_output = test_model.forward((tabular_data, image_data))
    # check output is correct size
    assert isinstance(test_output, list)
    assert len(test_output) == 1
    assert isinstance(test_output[0], torch.Tensor)
    assert test_output[0].shape == (1, 1)

    # wrong number of modalities
    with pytest.raises(ValueError, match=r"Wrong number of inputs for model!"):
        test_model.forward(tuple(tabular_data))
    with pytest.raises(ValueError, match=r"Wrong number of inputs for model!"):
        test_model.forward((tabular_data, tabular_data, image_data))
    with pytest.raises(TypeError, match=r"Wrong input type for model! Expected tuple"):
        test_model.forward(tabular_data)


def test_ImageChannelWiseMultiAttention():
    test_model = fusion_model_dict["ImageChannelWiseMultiAttention"]

    # attributes before initialising
    assert hasattr(test_model, "method_name")
    assert test_model.method_name == "Channel-wise Image attention"
    assert hasattr(test_model, "modality_type")
    assert test_model.modality_type == "tabular_image"
    assert hasattr(test_model, "fusion_type")
    assert test_model.fusion_type == "attention"

    # attributes after initialising
    test_model = test_model("binary", [15, None, (100, 100)], None)

    assert hasattr(test_model, "prediction_task")
    assert test_model.prediction_task == "binary"
    assert hasattr(test_model, "mod1_layers")
    assert hasattr(test_model, "img_layers")
    assert hasattr(test_model, "fused_layers")
    assert hasattr(test_model, "final_prediction")
    assert hasattr(test_model, "fused_dim")

    # test forward pass
    tabular_data = torch.rand((1, 15))
    image_data = torch.rand((1, 1, 100, 100))
    # forward pass
    test_output = test_model.forward((tabular_data, image_data))
    # check output is correct size
    assert isinstance(test_output, list)
    assert len(test_output) == 1
    assert isinstance(test_output[0], torch.Tensor)
    assert test_output[0].shape == (1, 1)

    # wrong number of modalities
    with pytest.raises(ValueError, match=r"Wrong number of inputs for model!"):
        test_model.forward(tuple(tabular_data))
    with pytest.raises(ValueError, match=r"Wrong number of inputs for model!"):
        test_model.forward((tabular_data, tabular_data, image_data))
    with pytest.raises(TypeError, match=r"Wrong input type for model! Expected tuple"):
        test_model.forward(tabular_data)


def test_CrossmodalMultiheadAttention():
    test_model = fusion_model_dict["CrossmodalMultiheadAttention"]

    # attributes before initialising
    assert hasattr(test_model, "method_name")
    assert test_model.method_name == "Crossmodal multi-head attention"
    assert hasattr(test_model, "modality_type")
    assert test_model.modality_type == "tabular_image"
    assert hasattr(test_model, "fusion_type")
    assert test_model.fusion_type == "attention"

    # attributes after initialising
    test_model = test_model("binary", [15, None, (100, 100)], None)

    assert hasattr(test_model, "prediction_task")
    assert test_model.prediction_task == "binary"
    assert hasattr(test_model, "mod1_layers")
    assert hasattr(test_model, "img_layers")
    assert hasattr(test_model, "attention")
    assert hasattr(test_model, "final_prediction")
    assert hasattr(test_model, "fused_dim")

    # test forward pass
    tabular_data = torch.rand((1, 15))
    image_data = torch.rand((1, 1, 100, 100))
    # forward pass
    test_output = test_model.forward((tabular_data, image_data))
    # check output is correct size
    assert isinstance(test_output, list)
    assert len(test_output) == 1
    assert isinstance(test_output[0], torch.Tensor)
    assert test_output[0].shape == (1, 1)

    # wrong number of modalities
    with pytest.raises(ValueError, match=r"Wrong number of inputs for model!"):
        test_model.forward(tuple(tabular_data))
    with pytest.raises(ValueError, match=r"Wrong number of inputs for model!"):
        test_model.forward((tabular_data, tabular_data, image_data))
    with pytest.raises(TypeError, match=r"Wrong input type for model! Expected tuple"):
        test_model.forward(tabular_data)


def test_ImageDecision():
    test_model = fusion_model_dict["ImageDecision"]

    # attributes before initialising
    assert hasattr(test_model, "method_name")
    assert test_model.method_name == "Image decision fusion"
    assert hasattr(test_model, "modality_type")
    assert test_model.modality_type == "tabular_image"
    assert hasattr(test_model, "fusion_type")
    assert test_model.fusion_type == "operation"

    # attributes after initialising
    test_model = test_model("binary", [15, None, (100, 100)], None)

    assert hasattr(test_model, "prediction_task")
    assert test_model.prediction_task == "binary"
    assert hasattr(test_model, "mod1_layers")
    assert test_model.mod1_layers['layer 1'][0].in_features == 15
    assert hasattr(test_model, "img_layers")
    assert hasattr(test_model, "final_prediction_tab1")
    assert hasattr(test_model, "final_prediction_img")
    assert hasattr(test_model, "forward")
    assert hasattr(test_model, "fusion_operation")
    # asserting that the default fusion operation is a mean with a simple calculation
    assert test_model.fusion_operation(torch.randn(8, 10), torch.randn(8, 10)).shape == torch.Size([8, 10])
    assert test_model.fusion_operation(torch.Tensor([1.0]), torch.Tensor([2.0])) == torch.Tensor([1.5])

    # test forward pass
    tabular_data = torch.rand((1, 15))
    image_data = torch.rand((1, 1, 100, 100))
    # forward pass
    test_output = test_model.forward((tabular_data, image_data))
    # check output is correct size
    assert isinstance(test_output, list)
    assert len(test_output) == 1
    assert isinstance(test_output[0], torch.Tensor)
    assert test_output[0].shape == (1, 1)

    # wrong number of modalities
    with pytest.raises(ValueError, match=r"Wrong number of inputs for model!"):
        test_model.forward(tuple(tabular_data))
    with pytest.raises(ValueError, match=r"Wrong number of inputs for model!"):
        test_model.forward((tabular_data, tabular_data, image_data))
    with pytest.raises(TypeError, match=r"Wrong input type for model! Expected tuple"):
        test_model.forward(tabular_data)


def test_ConcatImgLatentTabDoubleTrain():
    test_model = fusion_model_dict["ConcatImgLatentTabDoubleTrain"]

    # attributes before initialising
    assert hasattr(test_model, "method_name")
    assert test_model.method_name == "Pretrained Latent Image + Tabular Data"
    assert hasattr(test_model, "modality_type")
    assert test_model.modality_type == "tabular_image"
    assert hasattr(test_model, "fusion_type")
    assert test_model.fusion_type == "subspace"
    assert hasattr(test_model, "subspace_method")

    # attributes after initialising
    # modality 1: tabular 15 features
    # modality 2: image latent space 64 features
    test_model = test_model("binary", [15, 64, None], None)

    assert hasattr(test_model, "prediction_task")
    assert test_model.prediction_task == "binary"
    assert hasattr(test_model, "fused_layers")
    assert hasattr(test_model, "fused_dim")
    assert test_model.fused_dim == 15 + 64  # 15 from tabular data, rest from image after layers and flattening
    assert hasattr(test_model, "final_prediction")
    assert hasattr(test_model, "latent_dim")
    assert test_model.latent_dim == 64

    # test forward pass
    tabular_data = torch.rand((1, 15))
    image_latent = torch.rand((1, 64))
    # forward pass
    test_output = test_model.forward((tabular_data, image_latent))
    # check output is correct size
    assert isinstance(test_output, list)
    assert len(test_output) == 1
    assert isinstance(test_output[0], torch.Tensor)
    assert test_output[0].shape == (1, 1)

    # wrong number of modalities
    with pytest.raises(ValueError, match=r"Wrong number of inputs for model!"):
        test_model.forward(tuple(tabular_data))
    with pytest.raises(ValueError, match=r"Wrong number of inputs for model!"):
        test_model.forward((tabular_data, tabular_data, image_latent))
    with pytest.raises(TypeError, match=r"Wrong input type for model! Expected tuple"):
        test_model.forward(tabular_data)


def test_ConcatImgLatentTabDoubleLoss():
    test_model = fusion_model_dict["ConcatImgLatentTabDoubleLoss"]

    # attributes before initialising
    assert hasattr(test_model, "method_name")
    assert test_model.method_name == "Trained Together Latent Image + Tabular Data"
    assert hasattr(test_model, "modality_type")
    assert test_model.modality_type == "tabular_image"
    assert hasattr(test_model, "fusion_type")
    assert test_model.fusion_type == "subspace"

    # attributes after initialising
    # modality 1: tabular data 15 features
    # modality 2: image data 100x100 2D

    test_model = test_model("binary", [15, None, (100, 100)], None)

    assert hasattr(test_model, "prediction_task")
    assert test_model.prediction_task == "binary"
    assert hasattr(test_model, "fused_layers")
    assert hasattr(test_model, "final_prediction")
    assert hasattr(test_model, "fused_dim")
    assert hasattr(test_model, "custom_loss")
    assert hasattr(test_model, "latent_dim")
    assert hasattr(test_model, "new_encoder")
    assert hasattr(test_model, "new_decoder")

    # test forward pass
    tabular_data = torch.rand((1, 15))
    image_data = torch.rand((1, 1, 100, 100))
    # forward pass
    test_output = test_model.forward((tabular_data, image_data))
    # check output is correct size
    assert isinstance(test_output, list)
    assert len(test_output) == 2  # we've got reconstructions now too
    assert isinstance(test_output[0], torch.Tensor)
    assert test_output[0].shape == (1, 1)
    assert isinstance(test_output[1], torch.Tensor)
    assert test_output[1].shape == (1, 1, 100, 100)

    # wrong number of modalities
    with pytest.raises(ValueError, match=r"Wrong number of inputs for model!"):
        test_model.forward(tuple(tabular_data))
    with pytest.raises(ValueError, match=r"Wrong number of inputs for model!"):
        test_model.forward((tabular_data, tabular_data, image_data))
    with pytest.raises(TypeError, match=r"Wrong input type for model! Expected tuple"):
        test_model.forward(tabular_data)


def test_DAETabImgMaps():
    test_model = fusion_model_dict["DAETabImgMaps"]

    # attributes before initialising
    assert hasattr(test_model, "method_name")
    assert test_model.method_name == "Denoising tabular autoencoder with image maps"
    assert hasattr(test_model, "modality_type")
    assert test_model.modality_type == "tabular_image"
    assert hasattr(test_model, "fusion_type")
    assert test_model.fusion_type == "subspace"
    assert hasattr(test_model, "subspace_method")

    # attributes after initialising
    # 1 modality: concatenated denoise tabular data and image maps

    test_model = test_model("binary", [850, None, None], None)

    assert hasattr(test_model, "prediction_task")
    assert test_model.prediction_task == "binary"
    assert hasattr(test_model, "fusion_layers")
    assert hasattr(test_model, "final_prediction")

    # test forward pass
    tabular_data = torch.rand((1, 850))

    # forward pass
    test_output = test_model.forward(tabular_data)

    # check output is correct size
    assert isinstance(test_output, list)
    assert len(test_output) == 1
    assert isinstance(test_output[0], torch.Tensor)
    assert test_output[0].shape == (1, 1)

    # wrong number of modalities
    with pytest.raises(TypeError, match=r"Wrong input type for model! Expected torch.Tensor"):
        test_model.forward([torch.randn(8, 25)])
