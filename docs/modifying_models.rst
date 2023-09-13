Modifying the fusion models
===========================

- You can modify the fusion models by passing in a dictionary of the attributes you want to modify into the `get_data` and `training` functions.
- The attributes that can be modified are listed below with some guidance on how they can be changed.
- You can modify the layers in various fusion models or attributes such as custom loss function or latent dimension.
- There's an example in the examples section showing modification of the fusion models.

.. warning::
  If you change the layers in the fusion models such as :attr:`mod1_layers` or :attr:`img_layers`, you may also need to change the attribute :attr:`fused_layers` where appropriate.
  This is because the :attr:`fused_layers` attribute is used in the `forward` method of the fusion models to calculate the fused features and usually follows directly on
  from the :attr:`mod1_layers` or :attr:`img_layers` attributes. For example, if you change the final layer in the :attr:`mod1_layers` attribute of the :class:`~.Tabular1Unimodal` model to 
  output 100 features, you will also need to change the :attr:`fused_layers` attribute to have an input of 100 features. 

  The :func:`~.check_model_validity.check_fused_layers` function rewrites the first layer of the fused_layers to be the models :attr:`fused_dim` attribute and is run 
  in the model's :func:`calc_fused_layers` method. So if you change the :attr:`fused_dim` attribute to the correct number for the method as well as the other layers, 
  then the model should run as expected.


Modifiable attributes of the fusion models
------------------------------------------


:class:`.ConcatImgLatentTabDoubleLoss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. list-table:: 
  :widths: 40 60
  :header-rows: 1
  :stub-columns: 1

  * - Attribute
    - Guidance
  * - :attr:`~.ConcatImgLatentTabDoubleLoss.latent_dim`
    - int
  * - :attr:`~.ConcatImgLatentTabDoubleLoss.encoder`
    - ``nn.Sequential``
  * - :attr:`~.ConcatImgLatentTabDoubleLoss.decoder`
    - ``nn.Sequential``
  * - :attr:`~.ConcatImgLatentTabDoubleLoss.custom_loss`
    - Loss function e.g. ``nn.MSELoss``
  * - :attr:`~.ConcatImgLatentTabDoubleLoss.fused_layers`
    - ``nn.Sequential``



.. list-table:: Modifiable attributes of the fusion models
  :widths: 20 20 60
  :header-rows: 1
  :stub-columns: 1

  * - Method
    - Attribute
    - Guidance
  * - all
    - ``mod1_layers``
    - 
      * ``nn.ModuleDict``
      * Changes ``mod1_layers`` for all fusion models that have that attribute.
      * Modification overridden if ``mod1_layers`` is modified for a specific model.
  * - 
    - ``mod2_layers``
    - 
      *  ``nn.ModuleDict``
      * Changes ``mod2_layers`` for all fusion models that have that attribute.
      * Modification overridden if ``mod2_layers`` is modified for a specific model.
  * -
    - ``img_layers``
    - 
      * ``nn.ModuleDict``
      * Changes ``img_layers`` for all fusion models that have that attribute.
      * Modification overridden if ``img_layers`` is modified for a specific model.
  * - :class:`.ConcatImgLatentTabDoubleLoss`
    - :attr:`~.ConcatImgLatentTabDoubleLoss.latent_dim`
    - int
  * -
    - :attr:`~.ConcatImgLatentTabDoubleLoss.encoder`
    - ``nn.Sequential``
  * -
    - :attr:`~.ConcatImgLatentTabDoubleLoss.decoder`
    - ``nn.Sequential``
  * -
    - :attr:`~.ConcatImgLatentTabDoubleLoss.custom_loss`
    - Loss function e.g. ``nn.MSELoss``
  * -
    - :attr:`~.ConcatImgLatentTabDoubleLoss.fused_layers`
    - ``nn.Sequential``
  * - :class:`.ConcatImgLatentTabDoubleTrain`
    - :attr:`~.ConcatImgLatentTabDoubleTrain.fused_layers`
    - ``nn.Sequential``
  * - :class:`.concat_img_latent_tab_subspace_method`
    - :attr:`.autoencoder.latent_dim`
    - int
  * -
    - :attr:`.autoencoder.encoder`
    - ``nn.Sequential``
  * -
    - :attr:`.autoencoder.decoder`
    - ``nn.Sequential``
  * - :class:`.ConcatImageMapsTabularData`
    - :attr:`~.ConcatImageMapsTabularData.img_layers`
    - 
      * ``nn.Sequential``
      * Overrides modification of ``img_layers`` made to "all"
  * - :class:`.ConcatImageMapsTabularMaps`
    - :attr:`~.ConcatImageMapsTabularMaps.mod1_layers`
    - 
      * ``nn.Sequential``
      * Overrides modification of ``mod1_layers`` made to "all"
  * -
    - :attr:`~.ConcatImageMapsTabularMaps.img_layers`
    - 
      * ``nn.Sequential``
      * Overrides modification of ``img_layers`` made to "all"
  * -
    - :attr:`~.ConcatImageMapsTabularMaps.fused_layers`
    - ``nn.Sequential``
  * - :class:`.ConcatTabularData`
    - :attr:`~.ConcatTabularData.mod1_layers`
    - 
      * ``nn.Sequential``
      * Overrides modification of ``mod1_layers`` made to "all"
  * -
    - :attr:`~.ConcatTabularData.mod2_layers`
    -  
      * ``nn.Sequential``
      * Overrides modification of ``mod2_layers`` made to "all"
  * -
    - :attr:`~.ConcatTabularData.fused_layers`
    - ``nn.Sequential``
  * - :class:`.ConcatTabularFeatureMaps`
    - :attr:`~.ConcatTabularFeatureMaps.mod1_layers`
    - 
      * ``nn.Sequential``
      * Overrides modification of ``mod1_layers`` made to "all"
  * -
    - :attr:`~.ConcatTabularFeatureMaps.mod2_layers`
    - 
      * ``nn.Sequential``
      * Overrides modification of ``mod2_layers`` made to "all"
  * -
    - :attr:`~.ConcatTabularFeatureMaps.fused_layers`
    - ``nn.Sequential``
  * - :class:`.CrossmodalMultiheadAttention`
    - :attr:`~.CrossmodalMultiheadAttention.attention_embed_dim`
    - int
  * -
    - :attr:`~.CrossmodalMultiheadAttention.mod1_layers`
    - 
      *  ``nn.Sequential``
      * Overrides modification of ``mod1_layers`` made to "all"
      * The total number of layers in ``mod1_layers`` must be equal to total number of layers in ``img_layers``.
  * -
    - :attr:`~.CrossmodalMultiheadAttention.img_layers`
    - 
      * ``nn.Sequential``
      * Overrides modification of ``img_layers`` made to "all"
      * The total number of layers in ``mod1_layers`` must be equal to total number of layers in ``img_layers``.
  * - :class:`.DAETabImgMaps`
    - :attr:`~.DAETabImgMaps.fusion_layers`
    - 
      * ``nn.Sequential``
      * The first layer's input features should be the number of tabular features, but if not then this is corrected in :meth:`~DAETabImgMaps.calc_fused_layers`
  * - :class:`.denoising_autoencoder_subspace_method`
    - :attr:`.autoencoder.latent_dim`
    - int
  * -
    - :attr:`.autoencoder.upsampler`
    - ``nn.Sequential``
  * -
    - :attr:`.autoencoder.downsampler`
    - ``nn.Sequential``
  * -
    - :attr:`~.img_unimodal.img_layers`
    - 
      * ``nn.Sequential``
      * Overrides modification of ``img_layers`` made to "all"
  * - :class:`.EdgeCorrGNN`
    - :attr:`~.EdgeCorrGNN.graph_conv_layers`
    -  
      * ``nn.Sequential`` of ``torch_geometric.nn.GCNConv`` Layers.
      * The first layer's input features should be the number of the second tabular modality's features, but if not then this is corrected.
  * -
    - :attr:`~.EdgeCorrGNN.dropout_prob`
    - Float between (not including) 0 and 1.
  * - :class:`.EdgeCorrGraphMaker`
    - :attr:`~.EdgeCorrGraphMaker.threshold`
    - Float between (not including) 0 and 1.
  * - :class:`.ImageChannelWiseMultiAttention`
    - :attr:`~.ImageChannelWiseMultiAttention.mod1_layers`
    - 
      *  ``nn.ModuleDict``
      * Overrides modification of ``mod1_layers`` made to "all"
      * Must have same number of layers as :attr:`~.ImageChannelWiseMultiAttention.img_layers`
  * -
    - :attr:`~.ImageChannelWiseMultiAttention.img_layers`
    - 
      * ``nn.ModuleDict``
      * Overrides modification of ``mod1_layers`` made to "all"
      * Must have same number of layers as :attr:`~.ImageChannelWiseMultiAttention.mod1_layers`
  * - :class:`.ImageDecision`
    - :attr:`~.ImageDecision.mod1_layers`
    - 
      *  ``nn.ModuleDict``
      * Overrides modification of ``mod1_layers`` made to "all"
  * -
    - :attr:`~.ImageDecision.img_layers`
    - 
      * ``nn.ModuleDict``
      * Overrides modification of ``img_layers`` made to "all"
  * -
    - :attr:`~.ImageDecision.fusion_operation`
    - Lambda function (such as mean, median, etc.)
  * - :class:`.ImgUnimodal`
    - :attr:`~.ImgUnimodal.img_layers`
    - 
      * ``nn.ModuleDict``
      * Overrides modification of ``img_layers`` made to "all"
  * - :class:`.MCVAE_tab`
    - :attr:`~.MCVAE_tab.latent_space_layers`
    - 
      *  ``nn.ModuleDict``
      * Input channels of first layer should be the latent space size but this is also ensured in :meth:`~.MCVAE_tab.calc_fused_layers`
  * - :class:`.MCVAESubspaceMethod`
    - :attr:`~.MCVAESubspaceMethod.num_latent_dims`
    - int
  * - :class:`.TabularCrossmodalMultiheadAttention`
    - :attr:`~.TabularCrossmodalMultiheadAttention.attention_embed_dim`
    - int
  * -
    - :attr:`~.TabularCrossmodalMultiheadAttention.mod1_layers`
    - 
      *  ``nn.ModuleDict``
      * Overrides modification of ``mod1_layers`` made to "all"
      * Must have same number of layers as :attr:`~.TabularCrossmodalMultiheadAttention.mod2_layers`
  * -
    - :attr:`~.TabularCrossmodalMultiheadAttention.mod2_layers`
    - 
      * ``nn.ModuleDict``
      * Overrides modification of ``mod2_layers`` made to "all"
      * Must have same number of layers as :attr:`.TabularCrossmodalMultiheadAttention.mod1_layers`
  * - :class:`.Tabular1Unimodal`
    - :attr:`~.Tabular1Unimodal.mod1_layers`
    - 
      * ``nn.ModuleDict``
      * Overrides modification of ``mod1_layers`` made to "all"
  * - :class:`.Tabular2Unimodal`
    - :attr:`~.Tabular2Unimodal.mod2_layers`
    - 
      * ``nn.ModuleDict``
      * Overrides modification of ``mod2_layers`` made to "all"
  * - :class:`.TabularChannelWiseMultiAttention`
    - :attr:`~.TabularChannelWiseMultiAttention.mod1_layers`
    - 
      * ``nn.ModuleDict``
      * Overrides modification of ``mod1_layers`` made to "all"
      * Must have same number of layers as :attr:`~.TabularChannelWiseMultiheadAttention.mod2_layers`
  * -
    - :attr:`~.TabularChannelWiseMultiAttention.mod2_layers`
    - 
      * ``nn.ModuleDict``
      * Overrides modification of ``mod1_layers`` made to "all"
      * Must have same number of layers as :attr:`~.TabularChannelWiseMultiheadAttention.mod1_layers`
  * - :class:`.TabularDecision`
    - :attr:`~.TabularDecision.mod1_layers`
    - 
      * ``nn.ModuleDict``
      * Overrides modification of ``mod1_layers`` made to "all"
  * -
    - :attr:`~.TabularDecision.mod2_layers`
    - 
      * ``nn.ModuleDict``
      * Overrides modification of ``mod2_layers`` made to "all"
  * -
    - :attr:`~.TabularDecision.fusion_operation`
    - Function (such as mean, median, etc.). Should act on the 1st dimension.



How to pass into get_data and training as a dictionary:
--------------------------------------------------------

How to construct the dictionary:

- First keys must be the methods in the table above
- Second keys must be the attributes in the table above (e.g. autoencoder.latent_dim rather than just latent_dim)
- Value is the value you want to change the attribute to

.. note::
  You don't need to pass the layer modifications in the 'all' key again under a specific method key if you want to modify the layers for all fusion models.
  For example, if :attr:`mod1_layers` is modified under 'all', and that is how you want it to be for all fusion models, then you don't need to pass it again under a specific method key.
  However, if you want :attr:`mod1_layers` to be modified equally for all fusion models except one, then you can pass the modifications in the 'all' key and then override the modifications 
  for the specific fusion model.

  I have included all possible modifiable attribute in each specific method in the example below for completeness.

Here's an example of a dictionary which is modifying all of the modifiable attributes of the fusion models:

.. code-block:: python

    layer_modifications = {
        "all": {
            "mod1_layers": nn.ModuleDict(
                {
                    "layer 1": nn.Sequential(
                        nn.Linear(20, 32),
                        nn.ReLU(),
                    ),
                    "layer 2": nn.Sequential(
                        nn.Linear(32, 66),
                        nn.ReLU(),
                    ),
                    "layer 3": nn.Sequential(
                        nn.Linear(66, 128),
                        nn.ReLU(),
                    ),
                }
            ),
        }
    }