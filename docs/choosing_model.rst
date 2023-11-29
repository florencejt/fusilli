Trait-Driven Model Imports
============================

``fusilli`` offers various models for data fusion available in :mod:`fusilli.fusionmodels`.
Each model is adaptable to different degrees, and you can modify them as described in :ref:`modifying-models`.

Choosing a Model
----------------

You have several ways to select a model:

1. **Direct Import**: Import a specific model from :mod:`fusilli.fusionmodels` at the beginning of your script.
2. **Use Model Chooser Function**: Utilize :func:`fusilli.utils.model_chooser.import_chosen_fusion_models` to filter models based on specific criteria.

Model Chooser Function Usage
----------------------------

The :func:`fusilli.utils.model_chooser.import_chosen_fusion_models` function takes a dictionary of criteria. The keys represent criteria names, and the values are the desired criteria.

For instance, to fetch models capable of fusing tabular and image data, use:

.. code-block:: python

    from fusilli.utils.model_chooser import import_chosen_fusion_models

    criteria = {
        "modality_type": ["tab_img"],
    }

    models = import_chosen_fusion_models(criteria)

This function returns a list of models that fulfill the specified criteria. Access these models by indexing the list (e.g., ``models[0]``, ``models[1]``, etc.). It will also display a list of all models meeting that description.

Examples of Criteria
---------------------

- **Attention-Based Fusion for Tabular and Image Data**:

.. code-block:: python

    criteria = {
        "modality_type": ["tab_img"],
        "fusion_type": ["attention"],
    }

- **Operation- or Attention-Based Fusion for Tabular and Image Data; also Uni-Modal Benchmark Models**:

.. code-block:: python

    criteria = {
        "modality_type": ["tab_img", "tabular1", "img"],
        "fusion_type": ["operation", "attention"],
    }

- **Subspace-Based Fusion for Any Modality Type**:

.. code-block:: python

    criteria = {
        "fusion_type": ["subspace"],
        "modality_type": "all",
    }

- **Specific Models by Name: Tabular1Unimodal, Tabular2Unimodal, and ConcatTabularData**:

.. code-block:: python

    criteria = {
        "class_name": ["Tabular1Unimodal", "Tabular2Unimodal", "ConcatTabularData"],
    }