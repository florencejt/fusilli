**fusilli**: an introduction
==============================================================

What is fusilli?
----------------

Fusilli is a library for multimodal fusion. It's built on top of PyTorch and PyTorch Lightning and provides a simple interface for building and comparing multimodal fusion models.
It's designed to be easy to use, but also flexible enough to support a variety of fusion methods.

But what is multimodal fusion?
------------------------------

Multimodal fusion is the process of combining information from multiple modalities (e.g. text, image, audio) to make predictions.
For example, you might want to predict somebody's age. You could use a brain MRI, a blood test, and a questionnaire to make this prediction.
Each of these modalities provides different information about the person, and the idea is that you could combine them to make a better prediction than you could with any single modality.

But how do you combine them? There are a variety of ways to do this, and this library supports many of them, but not all of them.
If you find some more methods out in the wild, please have a go at implementing them and submitting a pull request!
Templates and specific guidance on how to do this are in :ref:`contributing`.

For a more detailed explanation of multimodal fusion, and a glimpse into some of the models that are out there, this `paper <https://iopscience.iop.org/article/10.1088/2516-1091/acc2fe/meta>`_ by Cui et al. is a great place to start [1].


Why would you want to use fusilli?
----------------------------------

🩻📈 **You have a dataset that contains multiple modalities.**

Either two types of tabular data or one type of tabular data and one type of image data. Ever thought that maybe they'd be more powerful together?
Fusilli can help you find out if multimodal fusion is right for you! ✨


🤔🆘 **You've looked at methods for multimodal fusion and thought "wow, that's a lot of code" and "wow, there are so many names for the same concept".**

*So* relatable. Fusilli provides a simple way for comparing multimodal fusion models without having to trawl through Google Scholar! ✨


😵‍💫🙌 **You've found a multimodal fusion method that you want to try out, but you're not sure how to implement it or it's not quite right for your data.**

Fusilli allows the users to modify existing methods, such as changing the architecture of the model, and provides templates for implementing new methods! ✨

-----

[1] Cui, C., Yang, H., Wang, Y., Zhao, S., Asad, Z., Coburn, L. A., Wilson, K. T., Landman, B. A., & Huo, Y. (2022).
Deep Multi-modal Fusion of Image and Non-image Data in Disease Diagnosis and Prognosis:
A Review (arXiv:2203.15588). arXiv. https://doi.org/10.48550/arXiv.2203.15588
