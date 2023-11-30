.. _fusion-model-explanations:

Fusion Model Guide
============================================

Below are explanations and diagrams explaining the fusion models available in this library.
Some of the models are inspired by papers in the literature, so links to the papers are provided
where appropriate.

.. figure:: _static/cui_diagram.jpeg
    :align: left
    :width: 100%
    :alt: Model fusion diagram

The diagram above shows the categorisation of the fusion models available in this library. This image has been taken from `Cui et al. (2023) <https://iopscience.iop.org/article/10.1088/2516-1091/acc2fe>`_.

The table below shows the categorisation of the models available in this library. It is important to note that some of the methods in this library can probably be categorised in more than one way. For example, the :class:`.ConcatImgLatentTabDoubleTrain` model can be considered both an subspace-based model and an operation-based model. This is because it uses an autoencoder to learn an image latent space, which is then concatenated with the tabular data. However, the model also uses an operation (concatenation) to combine the modalities. The categorisation of the models are a guide, rather than a strict rule.

.. list-table::
    :widths: 15 85
    :header-rows: 1

    * - Fusion type
      - Description
    * - Unimodal
      - These models use only one modality (e.g. tabular data or images) to make a prediction.
    * - Operation
      - Operation-based models use operations to combine the modalities. For example, concatenation, element-wise summation, and element-wise multiplication. These methods are easy to implement and are often used as baselines.
    * - Attention
      - Attention-based models use attention mechanisms to combine the modalities. Attention mechanisms work by learning a weight for each modality's importance, which is then used to combine the modalities.
    * - Subspace
      - Subspace-based models try to learn a joint latent space for the modalities. This can be done simply by looking at the correlations between the modalities, or by using a variational autoencoder (VAE) to learn the latent space through trying to reconstruct the modalities using a lower-dimensional representation.
    * - Graph
      - Graph-based models look at the interactions between nodes in a graph, where the edges can be learned as similarities between nodes, for example.
    * - Tensor
      - Tensor-based models use tensor operations to combine the modalities, such as outer products to find inter-modal and intra-modal interactions.



Operation-based
---------------

:class:`.ConcatTabularFeatureMaps`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tabular-tabular fusion model works by passing each tabular modality through its own set of fully-connected layers, and then concatenating the outputs of these layers together. The concatenated features are then passed through another set of fully-connected layers to make a prediction.

.. image:: _static/ConcatTabularFeatureMaps.png
    :align: left

------

:class:`.ConcatTabularData`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tabular-tabular fusion model works by concatenating the tabular data together, and then passing the concatenated features through a set of fully-connected layers to make a prediction.

.. image:: _static/ConcatTabularData.png
    :align: left

-----

:class:`.TabularDecision`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tabular-tabular fusion model works by passing each tabular modality through its own set of fully-connected layers to make a prediction for each modality. The two predictions are then averaged to make a final prediction. This is known as a "decision-level fusion" method.

.. image:: _static/TabularDecision.png
    :align: left

-----

:class:`.ConcatImageMapsTabularData`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tabular-image fusion model works by passing the image through a convolutional neural network (CNN) to extract features from the image. The tabular data is then concatenated with the image features, and the concatenated features are passed through a set of fully-connected layers to make a prediction.

.. image:: _static/ConcatImageMapsTabularData.png
    :align: left

-----

:class:`.ConcatImageMapsTabularMaps`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tabular-image fusion model works by passing the image through a CNN to extract features from the image. The tabular data is also passed through its own fully-connected layers to get tabular feature maps. The tabular features are then concatenated with the image features, and the concatenated features are passed through a set of fully-connected layers to make a prediction.


.. image:: _static/ConcatImageMapsTabularMaps.png
    :align: left

-----

:class:`.ImageDecision`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tabular-image fusion model works by passing each modality through its own network (fully-connected for tabular, CNN for image) to create their own predictions. The two predictions are then averaged to make a final prediction.
This is known as a "decision-level fusion" method.

.. image:: _static/ImageDecision.png
    :align: left

-----

:class:`.ActivationFusion`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is a tabular-tabular fusion model inspired by `Chen et al. (2022) <https://link.springer.com/article/10.1007/s00432-022-04180-1>`_ : *MDFNet: application of multimodal fusion method based on skin image and clinical data to skin cancer classification*.

The feature maps from the two separate modality networks are multiplied together, passed through a tanh activation function, then a sigmoid activation function, and then concatenated with the first tabular modality feature map.
The concatenated feature maps are then passed through a set of fully-connected layers to make a prediction.

.. note::

    A tabular-image version is coming soon!

.. image:: _static/ActivationFusion.png
    :align: left

-----

:class:`.AttentionAndActivation`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Again, this is a tabular-tabular fusion model inspired by `Chen et al. (2022) <https://link.springer.com/article/10.1007/s00432-022-04180-1>`_ : *MDFNet: application of multimodal fusion method based on skin image and clinical data to skin cancer classification*.

This one is an extension of :class:`.ActivationFusion`, where a self-attention module has been added to the second tabular modality's pipeline before its network layers.
The second tabular modality is passed through a fully-connected layer to be downsampled (downsampling factor can be modified), then a RelU, another upsampling fully-connected layer, and then a sigmoid activation function. Then the output is multipled by the original second tabular modality input data, and passed through its own fully-connected layers.
After this, the process is the same as :class:`.ActivationFusion`.

.. note::

    A tabular-image version is coming soon!

.. image:: _static/ActivationandSelfAttention.png
    :align: left



Attention-based
---------------

:class:`.TabularChannelWiseMultiAttention`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tabular-tabular fusion model works by passing each tabular modality through its own set of fully-connected layers.
At each layer, the feature maps from the first tabular modality are multiplied into the feature maps from the second tabular modality, effectively modulating the feature maps from the second modality with the feature maps from the first modality (an attention mechanism).
The final second tabular feature maps are then passed through a set of fully-connected layers to make a prediction.

This model is inspired by `Duanmu et al. (2020) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9821469/#:~:text=Deep%20learning%20using%20longitudinal%20multiparametric,%2C%20and%20mid%2Dtreatment%20adjustment.>`_ :  *Deep learning prediction of pathological complete response, residual cancer burden, and progression-free survival in breast cancer patients*.

.. image:: _static/TabularChannelwiseAttention.png
    :align: left

-----

:class:`.TabularCrossmodalMultiheadAttention`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tabular-tabular fusion model works by passing each tabular modality through its own set of fully-connected layers.
Self attention is applied to each modality, and then crossmodal attention is applied to the two modalities.
The output of the crossmodal attention is then passed through a fully-connected layer to make a prediction.

This model is inspired by MADDi - Multimodal Alzheimer's Disease Diagnosis Framework by `Golovankesky et al. (2022) <https://arxiv.org/abs/2206.08826>`_. They also have their own `code <https://github.com/rsinghlab/MADDi>`_ available.

.. image:: _static/TabularCrossmodalAttention.png
    :align: left

-----

:class:`.CrossmodalMultiheadAttention`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tabular-image fusion model works the same as the :class:`.TabularCrossmodalMultiheadAttention` model, except that the tabular modality is passed through a fully-connected layer, and the image modality is passed through a CNN.

.. image:: _static/CrossmodalMultiheadAttention.png
    :align: left

-----

:class:`.ImageChannelWiseMultiAttention`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tabular-image model works the same as the :class:`.TabularChannelWiseMultiAttention` model, except that the tabular modality is passed through a fully-connected layer, and the image modality is passed through a CNN.

.. image:: _static/ImageChannelwiseMultiheadAttention.png
    :align: left



Subspace-based
--------------

:class:`.MCVAE_tab`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This subspace-based model uses the Multi-channel Variational Autoencoder (MCVAE) by `Antelmi et al. (2019) <https://proceedings.mlr.press/v97/antelmi19a.html>`_.
This model works by passing each tabular modality as a separate 'channel' into a VAE with a modified loss function, which is then used to learn a joint latent space for the modalities.
The 1-dimensional joint latent space is then passed through a set of fully-connected layers to make a prediction.

For many more examples of multi-modal VAE-based models, I *highly recommend* looking at the Python library `Multi-view-AE <https://github.com/alawryaguila/multi-view-AE>`_ by Ana Aguila-Lawry et al.

.. image:: _static/MCVAE.png
    :align: left

-----

:class:`.ConcatImgLatentTabDoubleLoss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tabular-image model works by passing the image through an convolutional autoencoder to learn the latent space of the image.
The tabular data is concatenated with the image latent space, and the concatenated features are passed through a set of fully-connected layers to make a prediction.

The reconstruction loss of the autoencoder is added to the loss function of the model, to encourage the model to learn a good latent space for the image. This means that the image autoencoder and the prediction model are trained at the same time.

.. image:: _static/ImgLatentDoubleLoss.png
    :align: left

-----

:class:`.ConcatImgLatentTabDoubleTrain`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Very similar to the :class:`.ConcatImgLatentTabDoubleLoss` model, except that the image autoencoder is trained separately to the prediction model.

.. image:: _static/ImgLatentDoubleTrain.png
    :align: left

-----

:class:`.DAETabImgMaps`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This tabular-image fusion model is inspired by `Zhao et al. (2022) <https://pubmed.ncbi.nlm.nih.gov/36553200/>`_ : *A Multimodal Deep Learning Approach to Predicting Systemic Diseases from Oral Conditions*.

The tabular data is input into a denoising autoencoder, which is upsamples the tabular data and uses dropout at the beginning of the network to make the model more robust to noise and missing data (simulating a common problem in medical data).
The image data is passed through a CNN to make a prediction, to learn prediction-relevant features from the image. The final two convolutional layers of the CNN are then flattened and concatenated with the upsampled tabular data, and the concatenated features are passed through a set of fully-connected layers to make a prediction.

The denoising autoencoder and the image CNN are trained separately from the prediction model, and the final prediction model is trained on the concatenated features.

.. image:: _static/DAETabImgMaps.png
    :align: left


Tensor-based
------------

Incoming!

Graph-based
-----------
.. warning::
    ⚠️ It is not possible to use external test set data with graph-based fusion models. Trying to use a "from new data" method such as :meth:`.RealsVsPreds.from_new_data` will result in an error.

:class:`.EdgeCorrGNN`
~~~~~~~~~~~~~~~~~~~~~~~~~~

This graph structure of this tabular-tabular model is made by calculating the correlation between the first tabular modality's features, and using the correlation as the edge weights in a graph. If the correlations are less than a certain threshold (default of 0.8), the edge is removed from the graph.
The node features of the graph are the second tabular modality features. The graph is then passed through a graph neural network (GNN) to make a prediction.

.. image:: _static/EdgeCorrGNN.png
    :align: left

:class:`.AttentionWeightedGNN`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is a model inspired by method in `Bintsi et al. (2023) <https://arxiv.org/abs/2307.04639>`_ : *Multimodal brain age estimation using interpretable adaptive population-graph learning*.
In the paper, the method is based on adaptive graph learning. However, the fusilli implementation changes this to a static graph because the adaptive graph learning method is not yet implementable in fusilli.

The attention-weighted GNN works by pretraining a tabular-tabular fusion model, :meth:`.ConcatTabularData`, and then taking the "attention weights" from the model as the final input-sized layer output of the GNN.
These attention weights are multiplied with the concatenated first and second tabular modalities, and the Euclidean distance between each subject's attention-weighted features is calculated. If the distance between two subjects
is in the lowest 25% of all distances, an edge is created between the two subjects in the graph. The graph is then passed through a GNN to make a prediction.

.. image:: _static/AttentionWeightedGNN.png
    :align: left


-----

Unimodal
-----------

:class:`.Tabular1Unimodal`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A simple tabular model that uses a fully-connected network with the first tabular modality to make a prediction.

.. image:: _static/Tabular1Unimodal.png
    :align: left

-----

:class:`.Tabular2Unimodal`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A simple tabular model that uses a fully-connected network with the second tabular modality to make a prediction.

.. image:: _static/Tabular2Unimodal.png
    :align: left

-----

:class:`.ImgUnimodal`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A simple image model that uses a convolutional neural network (CNN) with the image modality to make a prediction.

.. image:: _static/ImageUnimodal.png
    :align: left