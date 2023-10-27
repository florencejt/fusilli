"""
Creating a subspace-based fusion model
=======================================

This tutorial will show you how to create a subspace-based fusion model.

.. note::

    I recommend looking at :ref:`how_to_contribute_a_template_other_fusion` before looking at this template, as I will skip over some of the details that are covered in that template (particularly regarding documentation and idiosyncrasies of the fusion model template).

There are **two** types of subspace-based fusion models in this library:

1. A model that has subspace methods trained **before** the main prediction model. An example of this is :class:`~fusilli.fusionmodels.tabularfusion.denoise_tab_img_maps.DAETabImgMaps`. This works by training the subspace method first, then using the output of the subspace method as the input to the main prediction model.
2. A model that has subspace methods (such as an autoencoder) trained **simultaneously** as the main prediction model. An example of this is :class:`~fusilli.fusionmodels.tabularimagefusion.concat_img_latent_tab_doubleloss.ConcatImgLatentTabDoubleLoss`. This works by implementing a joint loss function that combines the loss of the subspace method and the loss of the main prediction model.

We will look at how to create both of these types of models.
"""

# %%
# How to create a pre-trained subspace-based fusion model
# -------------------------------------------------------

# how to create it

# %%
# How to create a simultaneously-trained subspace-based fusion model
# ------------------------------------------------------------------

# how to create it
