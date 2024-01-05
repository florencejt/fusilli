Glossary
=========

This glossary provides definitions and explanations for key terms used in the Fusilli library. It aims to help users better understand the concepts and terminologies employed within the library.

.. glossary::

   fusion model
       A fusion model in Fusilli refers to a PyTorch ``nn.Module`` that implements a data fusion model. It is a deep learning model that takes in two data modalities and outputs a prediction.

   modality
       In multi-modal data fusion, a modality refers to a single data source. For example, in the context of fusing tabular data and image data, the tabular data is one modality and the image data is another modality.

   tabular data
       Tabular data is a data structure that is composed of rows and columns. Each row represents a single observation, and each column represents a single feature. Tabular data is also known as a data frame or a table.

   tabular1 and tabular2
        In the context of Fusilli, tabular1 and tabular2 refer to two tabular datasets that are to be fused together. The two tabular datasets must have the same number of rows, but can have different numbers of columns.

   WeightsAndBiases
        WeightsAndBiases is a machine learning experiment tracking tool that is used to track the performance of the models trained by Fusilli. It is used to log the training and validation losses, as well as the test metrics for the fusion models. For more information, see :ref:`wandb`.

