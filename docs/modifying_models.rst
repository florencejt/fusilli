.. _modifying-models:

Modify the Fusion Models
=================================

The fusion models in ``fusilli`` can be customized by passing a dictionary of attributes into the :func:`fusilli.data.prepare_fusion_data` and :func:`fusilli.train.train_and_save_models` functions.

Examples of how to modify the models can be found in the :ref:`advanced-examples` section.

Below are the modifiable attributes with guidance on how they can be changed:

.. note::
   If modifications are made to certain layers like :attr:`~.ActivationFusion.mod1_layers` or :attr:`~.AttentionAndActivation.img_layers`, the attribute :attr:`~.ActivationFusion.fused_layers` will be updated to ensure the first layer has the correct input features corresponding to the modified layers. Similarly, altering :attr:`~.ActivationFusion.fused_layers` will adjust the `final_prediction` layer's input features accordingly.

.. warning::
   Errors may occur if the input features of certain layer groups (e.g., :attr:`~.ActivationFusion.mod1_layers` and :attr:`~.ActivationFusion.img_layers`) are incorrect. For instance, changing :attr:`~.ActivationFusion.mod1_layers` input features to 20 while the actual number for the first tabular modality is 10 will result in a matrix multiplication error during the `forward` method.

.. warning::
   If you're using external test data, don't forget to pass the layer modifications into your evaluation figure function (like :func:`fusilli.eval.RealsVsPreds.from_new_data`).


Constructing the Layer Modification Dictionary
--------------------------------------------------------

To construct the dictionary:

- First keys should be the methods mentioned below.
- Second keys should be the attributes from the tables below.
- Value is the intended modification for the attribute.

For instance, modifying models using the "all" key applies those changes to all fusion models unless specifically overridden by a modification to a specific fusion model. Here's an example demonstrating this:

.. code-block:: python

    layer_modifications = {
        "all": {
            "mod1_layers": nn.ModuleDict(
                {
                    "layer 1": nn.Sequential(
                        nn.Linear(20, 32),
                        nn.ReLU(),
                    ),
                    # ... (additional layer modifications)
                }
            ),
        },  # end of "all" key
        "ConcatImgMapsTabularMaps": {  # overrides modifications made to "all"
            "mod1_layers": nn.ModuleDict(
                {
                    "layer 1": nn.Sequential(
                        nn.Linear(20, 100),
                        nn.ReLU(),
                    ),
                    # ... (additional layer modifications)
                }
            ),
        },
        # ... (additional fusion model modifications)
    }

------

Modifiable Attributes
---------------------

:class:`.ActivationFusion`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
  :widths: 40 60
  :header-rows: 1
  :stub-columns: 0

  * - Attribute
    - Guidance
  * - :attr:`~.ActivationFusion.mod1_layers`
    - ``nn.ModuleDict``
  * - :attr:`~.ActivationFusion.mod2_layers`
    - ``nn.ModuleDict``
  * - :attr:`~.ActivationFusion.fused_layers`
    - ``nn.Sequential``

:class:`.AttentionAndActivation`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
  :widths: 40 60
  :header-rows: 1
  :stub-columns: 0

  * - Attribute
    - Guidance
  * - :attr:`~.AttentionAndActivation.mod1_layers`
    - ``nn.ModuleDict``
  * - :attr:`~.AttentionAndActivation.mod2_layers`
    - ``nn.ModuleDict``
  * - :attr:`~.AttentionAndActivation.fused_layers`
    - ``nn.Sequential``
  * - :attr:`~.AttentionAndActivation.attention_reduction_ratio`
    - int


:class:`.AttentionWeightedGNN`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
  :widths: 40 60
  :header-rows: 1
  :stub-columns: 0

  * - Attribute
    - Guidance
  * - :attr:`~.AttentionWeightedGNN.graph_conv_layers`
    - ``nn.Sequential`` of ``torch_geometric.nn`` Layers.
  * - :attr:`~.AttentionWeightedGNN.dropout_prob`
    - Float between (not including) 0 and 1.


:class:`.AttentionWeightedGraphMaker`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
  :widths: 40 60
  :header-rows: 1
  :stub-columns: 0

  * - Attribute
    - Guidance
  * - :attr:`~.AttentionWeightedGraphMaker.early_stop_callback`
    - ``EarlyStopping`` object from ``from lightning.pytorch.callbacks import EarlyStopping``
  * - :attr:`~.AttentionWeightedGraphMaker.edge_probability_threshold`
    - Integer between 0 and 100.
  * - :attr:`~.AttentionWeightedGraphMaker.attention_MLP_test_size`
    - Float between 0 and 1.
  * - :attr:`~.AttentionWeightedGraphMaker.AttentionWeightingMLPInstance.weighting_layers`
    - ``nn.ModuleDict``: final layer output size must be the same as the input layer input size.
  * - :attr:`~.AttentionWeightedGraphMaker.AttentionWeightingMLPInstance.fused_layers`
    - ``nn.Sequential``



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
    - ``nn.ModuleDict``
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
    - ``nn.ModuleDict``
  * - :attr:`~.ConcatImageMapsTabularMaps.img_layers`
    - ``nn.ModuleDict``
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
    - ``nn.ModuleDict``
  * - :attr:`~.ConcatTabularFeatureMaps.mod2_layers`
    - ``nn.ModuleDict``
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
    - ``nn.ModuleDict``
  * - :attr:`~.CrossmodalMultiheadAttention.img_layers`
    - ``nn.ModuleDict``

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
      * ``nn.ModuleDict``
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