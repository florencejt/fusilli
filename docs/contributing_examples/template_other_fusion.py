"""
How to create your own fusion model
===============================================

I want to create my own fusion model! Does this sound like you? Then this is the template for you! ✨✨✨

.. note::

    **Is this the correct template for you?**

    If you want to implement a graph-based or subspace-based fusion model, please refer to the other templates.

    You'll know if you need to use them if the input into the model you're implementing can't be represented as a tuple of tensors of the original input data (modality1, modality2).

    For example:

    * If you're implementing a graph-based fusion model, the input into the model is a graph, not a tuple of tensors.
    * If you're implementing a subspace-based fusion model, the input into the model might be a latent space from a VAE trained on the original input data, not the original input data itself.

"""

# %%
# Step 1: Importing the libraries
# --------------------------------
# Let's import the libraries we need to create our model. Because we're using PyTorch, we need to import the PyTorch libraries
# as well as the :class:`~.ParentFusionModel` class and functions to help with checking model conditions and validity in the :mod:`~.utils.check_model_validity` module.

import torch.nn as nn
import torch

# importing the parent fusion model class
from fusilli.fusionmodels.base_model import ParentFusionModel

# importing functions to help with checking model conditions and validity
from fusilli.utils import check_model_validity


# %%
# Step 2: Creating the model structure
# ------------------------------------

# %%
# **Step 2.1: Creating the class**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Let's create the class for our model. We'll call it ``TemplateFusionModel``. This class will inherit from the
# :class:`~.ParentFusionModel` class and the :class:`~torch.nn.Module` class. This is because we want to inherit the
# methods and attributes from the :class:`~.ParentFusionModel` class and we want to make sure that our model is a
# PyTorch model.
#
# :class:`~.ParentFusionModel` has 3 input arguments:
#
# * ``pred_type`` : a string telling the model what type of prediction to perform. This is specified by the user in their python script or notebook.
# * ``data_dims`` : a list of the dimensions of the input data. This is calculated by :func:`~fusilli.data.prepare_fusion_data`.
# * ``params`` : a dictionary containing the parameters of the model. This is specified by the user in their python script or notebook.
#
# These input arguments have to be passed into the ``__init__()`` function of our fusion model. When running this library, this is done automatically for you in
# the :func:`~fusilli.train.train_and_save_models` function.


class TemplateFusionModel(ParentFusionModel, nn.Module):
    def __init__(self, pred_type, data_dims, params):
        ParentFusionModel.__init__(self, pred_type, data_dims, params)

    def forward(self, x):
        pass


# %%
# **Step 2.2: Setting the model attributes**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Each model has to have the following attributes at the class level (i.e. outside of the ``__init__()`` function and accessable without having to call ``TemplateFusionModel()``):
#
# * ``method_name`` : a string of the method name. This can be a better description of the method than the class name. For example, the class name might be ``ConcatTabularData`` but the method name might be ``Concatenation of tabular data``.
# * ``modality_type`` : a string containing the type of modality, which is one of the following: ``tabular1``, ``tabular2``, ``tabular_tabular``, ``tab_img``, ``img``.
# * ``fusion_type`` : a string containing the type of fusion, which is one of the following: ``operation``, ``attention``, ``tensor``, ``graph``, ``subspace``. To find out more about the different types of fusion, please refer to the :ref:`fusion-model-explanations` section.
#
# .. note::
#
#   The comment above the class attributes lets the attributes be documented automatically by Sphinx. This is why the comment is formatted in a specific way.

class TemplateFusionModel(ParentFusionModel, nn.Module):
    # str: name of the method
    method_name = "Template fusion model"
    # str: modality type
    modality_type = "tabular_tabular"  # or "tabular1", "tabular2", "tabular_tabular", "tabular_image", "img"
    # str: fusion type
    fusion_type = "attention"  # or "operation", "tensor", "graph", "subspace"

    def __init__(self, pred_type, data_dims, params):
        ParentFusionModel.__init__(self, pred_type, data_dims, params)

    def forward(self, x):
        pass


# %%
# **Step 2.3: Setting the model layers**
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now we need to set the layers of the model. This is done in the ``__init__()`` function of the model.
#
# There are two ways to set the layers of the model:
#
# 1. You can use the preset layers in the :class:`~.ParentFusionModel` class. This is the easiest way to create your own fusion model. You can see an example of this in the :class:`~fusilli.fusionmodels.tabularfusion.concat_data.ConcatTabularData` class.
# 2. You can create your own layers. This is the most flexible way to create your own fusion model but it might mean that the model is less easily comparible to other models in the library.
#
# Let's go through each of these methods in turn.
#
# **Method 1: Using preset layers**
#
# Let's say we want to use the preset layers in the :class:`~.ParentFusionModel` class. We can do this by calling the following functions:
#
# * :func:`~.set_mod1_layers` : sets the layers for the first tabular modality as ``self.mod1_layers``.
# * :func:`~.set_mod2_layers` : sets the layers for the second tabular modality as ``self.mod2_layers``.
# * :func:`~.set_img_layers` : sets the layers for the image modality as ``self.img_layers``.
# * :func:`~.set_fused_layers` : sets some layers that take place after the fusion of the modalities (may not be applicable for all fusion models) as ``self.fused_layers``. For example, if you're concatenating feature maps from multiple modalities, the fused layers would be the layers after the concatenation and before the prediction.
# * :func:`~.set_final_pred_layers` : sets the layers for the final prediction as ``self.final_predction``. We must set ``self.pred_type`` to the ``pred_type`` input argument of the ``__init__()`` function before calling this function. This is because the final prediction layers depend on the type of prediction we want to perform.
#
# .. note::
#   Calling ``self.set_mod1_layers()`` by itself is equivalent to calling ``self.mod1_layers = self.set_mod1_layers()``. This is because the ``set_mod1_layers()`` function assigns the layers to the ``mod1_layers`` attribute in :class:`~.ParentFusionModel`, which our model inherits from.
#   The same is true for the other :class:`~.ParentFusionModel` functions: ``set_mod2_layers()``, ``set_img_layers()``, ``set_fused_layers()``, and ``set_final_pred_layers()``.
#
# **Method 2: Creating your own layers**
#
# This is simply done by creating a dictionary of layers and assigning it to the ``mod1_layers`` attribute of the model. For example:
#
# .. code-block:: python
#
#   self.mod1_layers = nn.ModuleDict({
#       "linear1": nn.Linear(10, 20),
#       "linear2": nn.Linear(20, 30),
#       "linear3": nn.Linear(30, 40),
#   })
#
# Let's create our own layers for our model. We'll use the preset layers in the :class:`~.ParentFusionModel` class and make a tabular-tabular fusion model.

class TemplateFusionModel(ParentFusionModel, nn.Module):
    # str: name of the method
    method_name = "Template fusion model"
    # str: modality type
    modality_type = "tabular_tabular"  # or "tabular1", "tabular2", "tabular_tabular", "tabular_image", "img"
    # str: fusion type
    fusion_type = "attention"  # or "operation", "tensor", "graph", "subspace"

    def __init__(self, pred_type, data_dims, params):
        ParentFusionModel.__init__(self, pred_type, data_dims, params)

        self.pred_type = pred_type

        self.set_mod1_layers()  # set the layers for the first tabular modality
        self.set_mod2_layers()  # set the layers for the second tabular modality

        # Calculate the "fused_dim": how many features are there after the fusion? For example:
        mod1_layers_output_dim = self.mod1_layers[-1][0].out_features
        mod2_layers_output_dim = self.mod2_layers[-1][0].out_features
        self.fused_dim = (
                mod1_layers_output_dim + mod2_layers_output_dim
        )

        self.set_fused_layers(
            fused_dim=self.fused_dim)  # set the fused layers with an input dimension of self.fused_dim

        self.set_final_pred_layers(
            input_dim=64)  # set the final prediction layers with an input dimension of 64 (output dimension of fused layers)

    def forward(self, x):
        pass


# %%
# Step 3: Setting up model to be modifiable
# ------------------------------------------
#
# Great! We've set up the model structure. Now we need to make sure that the model is modifiable.
#
# In order to do this, we need to make sure that the model can handle if parts of it are changed.
# For example, if the number of output nodes in the final layers of ``self.mod1_layers`` is changed,
# the layers after it have to be recalculated so that there isn't a dimension mismatch.
#
# We can do this by creating a function called ``calc_fused_layers()``. This function should be called at the end of the ``__init__()`` function and should
# contain all the checks that need to be performed to make sure that the modifications made to the model are valid.
# The function ``set_final_pred_layers()`` should be moved into this function since it relies on the outputs of modifiable layers before it.
#
# .. warning::
#   This function must be called ``calc_fused_layers()``.
#   This is because the function is called whenever a modification is made to the model in :func:`~.modify_model_architecture`.
#
#   If you call the function something else, it won't be called when a modification is made to the model and the model won't be modifiable.
#
# **The steps we are taking here are:**
#
# 1. Create a function called ``calc_fused_layers()``.
# 2. Recalculate ``self.fused_dim`` in the ``calc_fused_layers()`` function to update the fused dimension if the model is modified.
# 3. Add a check in the ``calc_fused_layers()`` function with :func:`~.check_model_validity.check_fused_layers` to make sure that the fused layers are valid. This changes the first fused layer to have the correct input dimension (if it's not already correct) and outputs the output dimension of the fused layers.
# 4. Move the ``set_final_pred_layers()`` function into the ``calc_fused_layers()`` function and use the input from the fused layers to set the final prediction layers.
# 5. Call the ``calc_fused_layers()`` function at the end of the ``__init__()`` function.
#
# .. note::
#
#   If calculating ``self.fused_dim`` is complicated, you can create a separate function called ``get_fused_dim()`` and call it in ``__init__()`` and in ``calc_fused_layers()``.

class TemplateFusionModel(ParentFusionModel, nn.Module):
    # str: name of the method
    method_name = "Template fusion model"
    # str: modality type
    modality_type = "tabular_tabular"  # or "tabular1", "tabular2", "tabular_tabular", "tabular_image", "img"
    # str: fusion type
    fusion_type = "attention"  # or "operation", "tensor", "graph", "subspace"

    def __init__(self, pred_type, data_dims, params):
        ParentFusionModel.__init__(self, pred_type, data_dims, params)

        self.pred_type = pred_type

        self.set_mod1_layers()  # set the layers for the first tabular modality
        self.set_mod2_layers()  # set the layers for the second tabular modality

        self.get_fused_dim()

        self.set_fused_layers(
            fused_dim=self.fused_dim)  # set the fused layers with an input dimension of self.fused_dim

        self.calc_fused_layers()  # calculate the fused layers to make sure there aren't dimension mismatches

    def get_fused_dim(self):
        mod1_layers_output_dim = self.mod1_layers[-1][0].out_features
        mod2_layers_output_dim = self.mod2_layers[-1][0].out_features
        self.fused_dim = (
                mod1_layers_output_dim + mod2_layers_output_dim
        )

    def calc_fused_layers(self):
        self.get_fused_dim()

        self.fused_layers, out_dim = check_model_validity.check_fused_layers(
            self.fused_layers, self.fused_dim
        )

        self.set_final_pred_layers(
            input_dim=out_dim)  # set the final prediction layers with the output dimension of fused layers

    def forward(self, x):
        pass


# %%
# Step 4: Defining the forward function
# ----------------------------------------
# Let's define the forward function of our model. This is where we define how the data flows through the model. This example is concatenating the feature maps of two tabular modalities.
#
# **The input into the forward function is either:**
#
# * a tuple of tensors (modality1, modality2) if there are two modalities
# * a tensor of the original input data (if there is only one modality). This is probably not applicable to your model but it might be for a graph- or subspace-based fusion model.
#
# **The output of the forward function is a list containing the output of the model.**
# This is because some of the models in Fusilli output reconstructed data as well as the prediction, and this library is designed to handle this by all outputs either being a list of length 1 or 2.
#

def forward(self, x):
    x_tab1 = x[0]  # tabular1 data
    x_tab2 = x[1]  # tabular2 data

    # Passing the data through the modality layers
    for i, (k, layer) in enumerate(self.mod1_layers.items()):
        x_tab1 = layer(x_tab1)
        x_tab2 = self.mod2_layers[k](x_tab2)

    # Concatenating the feature maps from the two modalities
    out_fuse = torch.cat((x_tab1, x_tab2), dim=-1)
    # Passing the fused data through the fused layers
    out_fuse = self.fused_layers(out_fuse)

    # Passing the data through the final prediction layers
    out = self.final_prediction(out_fuse)

    return [
        out,
    ]


# %%
# Step 5: Adding checks
# ----------------------------
# Let's add some checks to make sure that the model components and the input data are what we expect them to be.
# We've already added checks to the ``self.fused_layers`` attribute in the ``calc_fused_layers()`` function.
# **The checks we are adding are:**
#
# * Checking that the input data is a tuple of tensors with :func:`~.check_model_validity.check_model_input`.
# * Checking that the modality layers are a :class:`~torch.nn.ModuleDict` with :func:`~.check_model_validity.check_dtype`.
#
# Your model might have more specific checks, such as checking that your modality layers have the same number of layers if that is a requirement of your model.
#
# At the beginning of the ``forward()`` function, we add the following check:

def forward(self, x):
    check_model_validity.check_model_input(x)

    # rest of forward function


# %%
# At the beginning of the ``calc_fused_layers()`` function, we add the following checks:

def calc_fused_layers(self):
    check_model_validity.check_dtype(self.mod1_layers, nn.ModuleDict, "mod1_layers")
    check_model_validity.check_dtype(self.mod2_layers, nn.ModuleDict, "mod2_layers")


# %%
# If we were using images, we would also add the following check at the beginning of the ``calc_fused_layers()`` function which checks that the image layers are a :class:`~torch.nn.ModuleDict` and that the image dimension is correct

def calc_fused_layers(self):
    check_model_validity.check_img_dim(self.img_layers, self.img_dim, "img_layers")


# %%
# Step 6: Adding documentation
# ----------------------------
# All that's left is to add documentation to the model. This is done by adding a docstring to the class and to the ``__init__()`` function.
# The docstring for the class should contain the following:
#
# * A description of the model.
# * The attributes of the model (all the attributes that start with ``self.``).
#
# The docstring for the ``__init__()`` function and other functions in the model (``calc_fused_layers()``, etc)should contain the following:
#
# * A description of the function.
# * The input arguments of the function.
# * The output of the function.
#
# .. note::
#   The docstrings are formatted in a specific way so that they can be documented automatically by Sphinx.
#
# Let's add documentation to our model and see it all come together!


class TemplateFusionModel(ParentFusionModel, nn.Module):
    """ Description of the model.

    More information about the model, perhaps a link to a paper, etc.

    Attributes
    ----------
    method_name : str
        Name of the method.
    modality_type : str
        Type of modality.
    fusion_type : str
        Type of fusion.
    pred_type : str
        Type of prediction to be performed.
    mod1_layers : dict
        Dictionary containing the layers of the first modality.
    mod2_layers : dict
        Dictionary containing the layers of the second modality.
    fused_dim : int
        Dimension of the fused layers.
    fused_layers : nn.Sequential
        Sequential layer containing the fused layers.
    final_prediction : nn.Sequential
        Sequential layer containing the final prediction layers. The final prediction layers
        take in the number of features of the fused layers as input.

    """

    # str: name of the method
    method_name = "Template fusion model"
    # str: modality type
    modality_type = "tabular_tabular"  # or "tabular1", "tabular2", "tabular_tabular", "tabular_image", "img"
    # str: fusion type
    fusion_type = "attention"  # or "operation", "tensor", "graph", "subspace"

    def __init__(self, pred_type, data_dims, params):
        """
        Initialising the model.

        Parameters
        ----------

        pred_type : str
            Type of prediction to be performed.
        data_dims : list
            List containing the dimensions of the data. This is calculated by :func:`~fusilli.data.prepare_fusion_data`.
        params : dict
            Dictionary containing the parameters of the model. This is specified by the user in their python script or notebook.
        """
        ParentFusionModel.__init__(self, pred_type, data_dims, params)
        self.pred_type = pred_type

        self.set_mod1_layers()  # set the layers for the first tabular modality
        self.set_mod2_layers()  # set the layers for the second tabular modality

        self.get_fused_dim()

        self.set_fused_layers(
            fused_dim=self.fused_dim)  # set the fused layers with an input dimension of self.fused_dim

        self.calc_fused_layers()  # calculate the fused layers to make sure there aren't dimension mismatches

    def get_fused_dim(self):
        """
        Get the number of input features of the fused layers.
        """
        mod1_layers_output_dim = self.mod1_layers[-1][0].out_features
        mod2_layers_output_dim = self.mod2_layers[-1][0].out_features
        self.fused_dim = (
                mod1_layers_output_dim + mod2_layers_output_dim
        )

    def calc_fused_layers(self):
        """
        Calculates the fused layers.
        """
        check_model_validity.check_dtype(self.mod1_layers, nn.ModuleDict, "mod1_layers")
        check_model_validity.check_dtype(self.mod2_layers, nn.ModuleDict, "mod2_layers")

        self.get_fused_dim()

        self.fused_layers, out_dim = check_model_validity.check_fused_layers(
            self.fused_layers, self.fused_dim
        )

        self.set_final_pred_layers(
            input_dim=out_dim)  # set the final prediction layers with the output dimension of fused layers

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : tuple
         Tuple containing the input data.

        Returns
        -------
        list
         List containing the output of the model.
        """
        check_model_validity.check_model_input(x)

        x_tab1 = x[0]  # tabular1 data
        x_tab2 = x[1]  # tabular2 data

        # Passing the data through the modality layers
        for i, (k, layer) in enumerate(self.mod1_layers.items()):
            x_tab1 = layer(x_tab1)
            x_tab2 = self.mod2_layers[k](x_tab2)

        # Concatenating the feature maps from the two modalities
        out_fuse = torch.cat((x_tab1, x_tab2), dim=-1)
        # Passing the fused data through the fused layers
        out_fuse = self.fused_layers(out_fuse)

        # Passing the data through the final prediction layers
        out = self.final_prediction(out_fuse)

        return [
            out,
        ]

# %%
# I hope this template has been helpful! If you have any questions, please feel free to ask in the GitHub Discussions page.
