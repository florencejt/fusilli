.. _modifying-models:

Modifying the fusion models
===========================

- You can modify the fusion models by passing in a dictionary of the attributes you want to modify into the `get_data` and `training` functions.
- The attributes that can be modified are listed below with some guidance on how they can be changed.
- You can modify the layers in various fusion models or attributes such as custom loss function or latent dimension.
- There's an example in the examples section showing modification of the fusion models.

.. note::
  If you change the layers in the fusion models such as :attr:`mod1_layers` or :attr:`img_layers`, then where appropriate the attribute :attr:`fused_layers` will be changed so 
  that the first layer has the correct number of input features (corresponding to the final output features of the modified layers). 

  Following on from this, if you change the attribute :attr:`fused_layers` then the ``final_prediction`` layer will be changed to have the correct number of input features
  (corresponding to the final output features of the modified :attr:`fused_layers`).


.. warning::
  An error will occur if the input features of the some layer groups (e.g. :attr:`mod1_layers` and :attr:`img_layers`) are not the correct size.
  For example, if you change the input features of :attr:`mod1_layers` to be 20, but the number of input features of the first tabular modality is 
  actually 10, then a matrix multiplication error will occur from the ``forward`` method.


Modifiable attributes of the fusion models
------------------------------------------


:class:`.ConcatImgLatentTabDoubleLoss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. list-table:: 
  :widths: 40 60
  :header-rows: 1
  :stub-columns: 0

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

------

:class:`.ConcatImgLatentTabDoubleTrain`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. list-table:: 
  :widths: 40 60
  :header-rows: 1
  :stub-columns: 0

  * - Attribute
    - Guidance
  * - :attr:`~.ConcatImgLatentTabDoubleTrain.fused_layers`
    - ``nn.Sequential``

------

:class:`.concat_img_latent_tab_subspace_method`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. list-table:: 
  :widths: 40 60
  :header-rows: 1
  :stub-columns: 0

  * - Attribute
    - Guidance
  * - :attr:`.autoencoder.latent_dim`
    - int
  * - :attr:`.autoencoder.encoder`
    - ``nn.Sequential``
  * - :attr:`.autoencoder.decoder`
    - ``nn.Sequential``

------

:class:`.ConcatImageMapsTabularData`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. list-table:: 
  :widths: 40 60
  :header-rows: 1
  :stub-columns: 0

  * - Attribute
    - Guidance
  * - :attr:`~.ConcatImageMapsTabularData.img_layers`
    - ``nn.Sequential``
  * - :attr:`~.ConcatImageMapsTabularData.fused_layers`
    - ``nn.Sequential``

------

:class:`.ConcatImageMapsTabularMaps`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. list-table:: 
  :widths: 40 60
  :header-rows: 1
  :stub-columns: 0

  * - Attribute
    - Guidance
  * - :attr:`~.ConcatImageMapsTabularMaps.mod1_layers`
    - ``nn.Sequential``
  * - :attr:`~.ConcatImageMapsTabularMaps.img_layers`
    - ``nn.Sequential``
  * - :attr:`~.ConcatImageMapsTabularMaps.fused_layers`
    - ``nn.Sequential``

------

:class:`.ConcatTabularData`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. list-table:: 
  :widths: 40 60
  :header-rows: 1
  :stub-columns: 0

  * - Attribute
    - Guidance
  * - :attr:`~.ConcatTabularData.fused_layers`
    - ``nn.Sequential``

------

:class:`.ConcatTabularFeatureMaps`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. list-table:: 
  :widths: 40 60
  :header-rows: 1
  :stub-columns: 0

  * - Attribute
    - Guidance
  * - :attr:`~.ConcatTabularFeatureMaps.mod1_layers`
    - ``nn.Sequential``
  * - :attr:`~.ConcatTabularFeatureMaps.mod2_layers`
    - ``nn.Sequential``
  * - :attr:`~.ConcatTabularFeatureMaps.fused_layers`
    - ``nn.Sequential``

------

:class:`.CrossmodalMultiheadAttention`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. list-table:: 
  :widths: 40 60
  :header-rows: 1
  :stub-columns: 0

  * - Attribute
    - Guidance
  * - :attr:`~.CrossmodalMultiheadAttention.attention_embed_dim`
    - int
  * - :attr:`~.CrossmodalMultiheadAttention.mod1_layers`
    - ``nn.Sequential``
  * - :attr:`~.CrossmodalMultiheadAttention.img_layers`
    - ``nn.Sequential``

------
  
:class:`.DAETabImgMaps`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. list-table:: 
  :widths: 40 60
  :header-rows: 1
  :stub-columns: 0

  * - Attribute
    - Guidance
  * - :attr:`~.DAETabImgMaps.fusion_layers`
    - ``nn.Sequential``

------

:class:`.denoising_autoencoder_subspace_method`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. list-table:: 
  :widths: 40 60
  :header-rows: 1
  :stub-columns: 0

  * - Attribute
    - Guidance
  * - :attr:`.autoencoder.latent_dim`
    - int
  * - :attr:`.autoencoder.upsampler`
    - ``nn.Sequential``
  * - :attr:`.autoencoder.downsampler`
    - ``nn.Sequential``
  * - :attr:`.img_unimodal.img_layers`
    - 
      * ``nn.Sequential``
      * Overrides modification of ``img_layers`` made to "all"
  * - :attr:`.img_unimodal.fused_layers`
    - ``nn.Sequential``

------

:class:`.EdgeCorrGNN`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. list-table:: 
  :widths: 40 60
  :header-rows: 1
  :stub-columns: 0

  * - Attribute
    - Guidance
  * - :attr:`~.EdgeCorrGNN.graph_conv_layers`
    -  
      * ``nn.Sequential`` of ``torch_geometric.nn.GCNConv`` Layers.
      * The first layer's input features should be the number of the second tabular modality's features, but if not then this is corrected.
  * - :attr:`~.EdgeCorrGNN.dropout_prob`
    - Float between (not including) 0 and 1.

------

:class:`.EdgeCorrGraphMaker`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: 
  :widths: 40 60
  :header-rows: 1
  :stub-columns: 0

  * - Attribute
    - Guidance
  * - :attr:`~.EdgeCorrGraphMaker.threshold`
    - Float between (not including) 0 and 1.

------

:class:`.ImageChannelWiseMultiAttention`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. list-table:: 
  :widths: 40 60
  :header-rows: 1
  :stub-columns: 0

  * - Attribute
    - Guidance
  * - :attr:`~.ImageChannelWiseMultiAttention.mod1_layers`
    -
      *  ``nn.ModuleDict``
      * Overrides modification of ``mod1_layers`` made to "all"
      * Must have same number of layers as :attr:`~.ImageChannelWiseMultiAttention.img_layers`
  * - :attr:`~.ImageChannelWiseMultiAttention.img_layers`
    - 
      * ``nn.ModuleDict``
      * Overrides modification of ``mod1_layers`` made to "all"
      * Must have same number of layers as :attr:`~.ImageChannelWiseMultiAttention.mod1_layers`
  * - :attr:`~.ImageChannelWiseMultiAttention.fused_layers`
    - ``nn.Sequential``

------

:class:`.ImageDecision`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. list-table:: 
  :widths: 40 60
  :header-rows: 1
  :stub-columns: 0

  * - Attribute
    - Guidance
  * - :attr:`~.ImageDecision.mod1_layers`
    - 
      *  ``nn.ModuleDict``
      * Overrides modification of ``mod1_layers`` made to "all"
  * - :attr:`~.ImageDecision.img_layers`
    - 
      * ``nn.ModuleDict``
      * Overrides modification of ``img_layers`` made to "all"
  * - :attr:`~.ImageDecision.fusion_operation`
    - Function (such as mean, median, etc.). Should act on the 1st dimension.

------

:class:`.ImgUnimodal`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: 
  :widths: 40 60
  :header-rows: 1
  :stub-columns: 0

  * - Attribute
    - Guidance
  * - :attr:`~.ImgUnimodal.img_layers`
    - 
      * ``nn.ModuleDict``
      * Overrides modification of ``img_layers`` made to "all"
  * - :attr:`~.ImgUnimodal.fused_layers`
    - ``nn.Sequential``

------

:class:`.MCVAE_tab`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. list-table::
  :widths: 40 60
  :header-rows: 1
  :stub-columns: 0

  * - Attribute
    - Guidance
  * - :attr:`~.MCVAE_tab.latent_space_layers`
    - 
      *  ``nn.ModuleDict``
      * Input channels of first layer should be the latent space size but this is also ensured in :meth:`~.MCVAE_tab.calc_fused_layers`
  * - :attr:`~.MCVAE_tab.fused_layers`
    - ``nn.Sequential``

------

:class:`.MCVAESubspaceMethod`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. list-table::
  :widths: 40 60
  :header-rows: 1
  :stub-columns: 0

  * - Attribute
    - Guidance
  * - :attr:`~.MCVAESubspaceMethod.num_latent_dims`
    - int

------

:class:`.TabularCrossmodalMultiheadAttention`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. list-table::
  :widths: 40 60
  :header-rows: 1
  :stub-columns: 0

  * - Attribute
    - Guidance
  * - :attr:`~.TabularCrossmodalMultiheadAttention.attention_embed_dim`
    - int
  * - :attr:`~.TabularCrossmodalMultiheadAttention.mod1_layers`
    - 
      *  ``nn.ModuleDict``
      * Overrides modification of ``mod1_layers`` made to "all"
      * Must have same number of layers as :attr:`~.TabularCrossmodalMultiheadAttention.mod2_layers`
  * - :attr:`~.TabularCrossmodalMultiheadAttention.mod2_layers`
    - 
      * ``nn.ModuleDict``
      * Overrides modification of ``mod2_layers`` made to "all"
      * Must have same number of layers as :attr:`.TabularCrossmodalMultiheadAttention.mod1_layers`

------

:class:`.Tabular1Unimodal`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. list-table::
  :widths: 40 60
  :header-rows: 1
  :stub-columns: 0

  * - Attribute
    - Guidance
  * - :attr:`~.Tabular1Unimodal.mod1_layers`
    - 
      * ``nn.ModuleDict``
      * Overrides modification of ``mod1_layers`` made to "all"
  * - :attr:`~.Tabular1Unimodal.fused_layers`
    - ``nn.Sequential``

------

:class:`.Tabular2Unimodal`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. list-table::
  :widths: 40 60
  :header-rows: 1
  :stub-columns: 0

  * - Attribute
    - Guidance
  * - :attr:`~.Tabular2Unimodal.mod2_layers`
    - 
      * ``nn.ModuleDict``
      * Overrides modification of ``mod2_layers`` made to "all"
  * - :attr:`~.Tabular2Unimodal.fused_layers`
    - ``nn.Sequential``

------

:class:`.TabularChannelWiseMultiAttention`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. list-table::
  :widths: 40 60
  :header-rows: 1
  :stub-columns: 0

  * - Attribute
    - Guidance
  * - :attr:`~.TabularChannelWiseMultiAttention.mod1_layers`
    - 
      * ``nn.ModuleDict``
      * Overrides modification of ``mod1_layers`` made to "all"
      * Must have same number of layers as :attr:`~.TabularChannelWiseMultiheadAttention.mod2_layers`
  * - :attr:`~.TabularChannelWiseMultiAttention.mod2_layers`
    - 
      * ``nn.ModuleDict``
      * Overrides modification of ``mod1_layers`` made to "all"
      * Must have same number of layers as :attr:`~.TabularChannelWiseMultiheadAttention.mod1_layers`
  * - :attr:`~.TabularChannelWiseMultiAttention.fused_layers`
    - ``nn.Sequential``

------

:class:`.TabularDecision`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. list-table::
  :widths: 40 60
  :header-rows: 1
  :stub-columns: 0

  * - Attribute
    - Guidance
  * - :attr:`~.TabularDecision.mod1_layers`
    - 
      * ``nn.ModuleDict``
      * Overrides modification of ``mod1_layers`` made to "all"
  * - :attr:`~.TabularDecision.mod2_layers`
    - 
      * ``nn.ModuleDict``
      * Overrides modification of ``mod2_layers`` made to "all"
  * - :attr:`~.TabularDecision.fusion_operation`
    - Function (such as mean, median, etc.). Should act on the 1st dimension.

------




How to pass into get_data and training as a dictionary:
--------------------------------------------------------

How to construct the dictionary:

- First keys must be the methods in the table above
- Second keys must be the attributes in the table above (e.g. autoencoder.latent_dim rather than just latent_dim)
- Value is the value you want to change the attribute to

.. note::

    **Note on modifying models using the "all" key:**

    Modifications under the 'all' key will be applied to all fusion models, unless specifically overwritten by a
    modification to a specific fusion model.

    For example, if you want to modify the attribute :attr:`mod1_layers` for every fusion model that uses it, then
    you can pass the :attr:`mod1_layers` in the 'all' key.

    If you want to modify the attribute :attr:`mod1_layers` for every fusion model except one, then you can pass the
    :attr:`mod1_layers` in the 'all' key and then override the modifications for the specific fusion model.

    An example on how to do this is shown below, where the modifications to :attr:`mod1_layers` under the 'all' key
    are overridden for the fusion model :class:`.ConcatImgMapsTabularMaps`.


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
        }, # end of "all" key
        "ConcatImgMapsTabularMaps": { # overrides the "mod1_layers" modifications made to "all"
            "mod1_layers": nn.ModuleDict(
                {
                    "layer 1": nn.Sequential(
                        nn.Linear(20, 100),
                        nn.ReLU(),
                    ),
                    "layer 2": nn.Sequential(
                        nn.Linear(100, 300),
                        nn.ReLU(),
                    ),
                    "layer 3": nn.Sequential(
                        nn.Linear(300, 250),
                        nn.ReLU(),
                    ),
                    "layer 4": nn.Sequential(
                        nn.Linear(250, 100),
                        nn.ReLU(),
                    ),
                }
            ),
        },
    }
