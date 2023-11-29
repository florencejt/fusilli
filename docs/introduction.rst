Introduction
==============================================================

What is fusilli?
----------------

Fusilli is a Python library for **multimodal data fusion**.

You can run fusion models on your own data for predicting whatever you want, and you can compare lots of different fusion methods easily to see which one works best.

Fusilli is built on top of PyTorch and PyTorch Lightning and provides a simple interface for building and comparing multimodal fusion models.
It's designed to be easy to use, but also flexible enough to support a variety of fusion methods.

What is multimodal data fusion?
---------------------------------

Multimodal fusion is the process of combining information from multiple modalities (e.g. text, image, audio) to make predictions.

For example, you might want to predict somebody's age. You could use a brain MRI, a blood test, and a questionnaire to make this prediction.
Each of these modalities provides different information about the person, and the idea is that you could combine them to make a better prediction than you could with any single modality.

But how do you combine them? There are a variety of ways to do this, and this library supports many of them, but not all of them.

For a more detailed explanation of multimodal data fusion, and a glimpse into some of the models that are out there, this `paper <https://iopscience.iop.org/article/10.1088/2516-1091/acc2fe/meta>`_ by Cui et al. is a great place to start [1] (not a #ad).


A note about contributing
---------------------------

If you find some more methods out in the wild, please have a go at implementing them and submitting a pull request!
Templates and specific guidance on how to do this are in :ref:`contributing`.

Contact
--------

If you have any questions, please feel free to contact me at florence.townend.21@ucl.ac.uk or on Twitter:
`@FlorenceTownend <https://twitter.com/florencetownend>`_.



-----

[1] Cui, C., Yang, H., Wang, Y., Zhao, S., Asad, Z., Coburn, L. A., Wilson, K. T., Landman, B. A., & Huo, Y. (2022).
Deep Multi-modal Fusion of Image and Non-image Data in Disease Diagnosis and Prognosis:
A Review (arXiv:2203.15588). arXiv. https://doi.org/10.48550/arXiv.2203.15588
