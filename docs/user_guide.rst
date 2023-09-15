User guide
=================


Input data
-----------------

Guidance on what data can be used with this module

Tabular
~~~~~~~~~~~~~

* Format: 
    * Tabular 1: csv file, study_id as index column name, pred_label as label column name
    * Tabular 1: csv file, study_id as index column name, pred_label as label column name

Example:

.. code-block:: python

    example here 

Tabular and Image
~~~~~~~~~~~~~~~~~~~

* Format: 
    * Tabular 1: csv file, study_id as index column name, pred_label as label column name
    * Image: pytorch pt file, items in same order as csv file.
        * when opening the pt file with ``torch.load()``, the dimensions should be *put in dimensions here*
    



Choosing a model
-----------------

Fusion Types
~~~~~~~~~~~~~

More info about the fusion types can be found on section :ref:`fusion_model_explanations`.

.. list-table:: 
    :widths: 20 80
    :header-rows: 1

    * - Fusion type
      - Description
    * - ``Uni-modal``
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
