.. _fusion-model-explanations:

Fusion Model Explanations
==========================

Below are explanations and diagrams explaining the fusion models available in this library.
Some of the models are inspired by papers in the literature, so links to the papers are provided
where appropriate.

.. list-table::
    :widths: 20 80
    :header-rows: 1

    * - Fusion type
      - Description
    * - ``unimodal``
      -
    * - ``operation``
      -
    * - ``attention``
      -
    * - ``subspace``
      -
    * - ``graph``
      -
    * - ``tensor``
      -


The categorisation of these methods is taken from Cui et al [link to paper here].

Operation-based
---------------

:class:`.ConcatTabularFeatureMaps`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: _static/ConcatTabularFeatureMaps.png
    :align: left

-----

:class:`.ConcatTabularData`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: _static/ConcatTabularData.png
    :align: left

-----

:class:`.TabularDecision`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: _static/TabularDecision.png
    :align: left

-----

:class:`.ConcatImageMapsTabularData`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: _static/ConcatImageMapsTabularData.png
    :align: left

-----

:class:`.ConcatImageMapsTabularMaps`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: _static/ConcatImageMapsTabularMaps.png
    :align: left

-----

:class:`.ImageDecision`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: _static/ImageDecision.png
    :align: left


Attention-based
---------------

:class:`.TabularChannelWiseMultiAttention`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: _static/TabularChannelwiseAttention.png
    :align: left

-----

:class:`.TabularCrossmodalMultiheadAttention`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: _static/TabularCrossmodalAttention.png
    :align: left

-----

:class:`.CrossmodalMultiheadAttention`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: _static/CrossmodalMultiheadAttention.png
    :align: left

-----

:class:`.ImageChannelWiseMultiAttention`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: _static/ImageChannelwiseMultiheadAttention.png
    :align: left



Subspace-based
--------------

:class:`.MCVAE_tab`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: _static/MCVAE.png
    :align: left

-----

:class:`.ConcatImgLatentTabDoubleLoss`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: _static/ImgLatentDoubleLoss.png
    :align: left

-----

:class:`.ConcatImgLatentTabDoubleTrain`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: _static/ImgLatentDoubleTrain.png
    :align: left

-----

:class:`.DAETabImgMaps`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: _static/DAETabImgMaps.png
    :align: left


Tensor-based
------------



Graph-based
-----------

:class:`.EdgeCorrGNN`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: _static/EdgeCorrGNN.png
    :align: left



-----

Unimodal
-----------

:class:`.Tabular1Unimodal`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: _static/Tabular1Unimodal.png
    :align: left

-----

:class:`.Tabular2Unimodal`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: _static/Tabular2Unimodal.png
    :align: left

-----

:class:`.ImgUnimodal`
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: _static/ImageUnimodal.png
    :align: left