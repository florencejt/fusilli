.. _data-loading:

Loading your Data
==================

``fusilli`` facilitates fusion of **tabular data with tabular data** or **tabular data with images**.

Data Format Requirements
----------------------------

Your data must adhere to specific formats for ``fusilli`` to read it correctly with the :func:`fusilli.data.prepare_fusion_data` function.

The paths to the data source files must be in a dictionary before being passed to the :func:`fusilli.data.prepare_fusion_data` function.

.. code-block:: python

    data_paths = {
        "tabular1": "path/to/tabular1_data.csv",
        "tabular2": "path/to/tabular2_data.csv",
        "image": "path/to/image_data.pt",
    }

.. warning::

    If you are not using a particular data source, set the value to ``""``.

    For example, if you are not using ``tabular2``, set ``tabular2`` in the dictionary to ``""``.

Tabular and Tabular Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

All tabular data must be in CSV format.

Columns named ``ID`` and ``prediction_label`` are required:

- ``ID``: Unique identifiers for each row.
- ``prediction_label``: Labels (integers for classification or floats for regression).


**Example of loading two tabular modalities:**

.. code-block:: python

    from fusilli.data import prepare_fusion_data

    data_paths = {
        "tabular1": "path/to/tabular1_data.csv",
        "tabular2": "path/to/tabular2_data.csv",
        "image": "",
    }

    data_module = prepare_fusion_data(prediction_task=...,
                                      fusion_model=some_example_model,
                                      data_paths=data_paths,
                                      output_paths=...)

Tabular and Image Data
~~~~~~~~~~~~~~~~~~~~~~~

Tabular data should follow the format specified above. Image data should be in a ``.pt`` file format with dimensions
``(num_samples, num_channels, height, width)``.

For example, for 100 2D 28x28 grey-scale images, my images.pt file would have the dimensions ``(100, 1, 28, 28)`` when I use ``torch.load()``.

For 100 3D 32x32x32 RGB images, my images.pt file would have the dimensions ``(100, 3, 32, 32, 32)`` when I use ``torch.load()``.

**Example of loading tabular and image data:**

.. code-block:: python

    from fusilli.data import prepare_fusion_data

    data_paths = {
        "tabular1": "path/to/tabular1_data.csv",
        "tabular2": "",
        "image": "path/to/image_data.pt",
    }

    data_module = prepare_fusion_data(prediction_task=...,
                                      fusion_model=some_example_model,
                                      data_paths=data_paths,
                                      output_paths=...)
Downsampling Images
*********************

To downsample images before model input, use the ``image_downsample_size`` parameter in the :func:`fusilli.data.prepare_fusion_data` function.

**Example of downsampling 2D images to 16x16:**

.. code-block:: python


    data_module = prepare_fusion_data(prediction_task=...,
                                      fusion_model=some_example_model,
                                      data_paths=data_paths,
                                      output_paths=...,
                                      image_downsample_size=(16, 16))


-----

Incorporating External Test Data
--------------------------------

For evaluating models with external test data:

- Provide paths to test data sources in another dictionary like ``data_paths`` with the same keys ``tabular1``, ``tabular2``, and ``image``.
- Use the same data format as the training data.

Calling the evaluation figures functions with the method ``from_new_data`` will evaluate the model on the external test data and plot the results.

**Example of training and evaluating a model with external test data:**

.. code-block:: python

    from fusilli.data import prepare_fusion_data
    from fusilli.train import train_and_save_models
    from fusilli.eval import RealsVsPreds

    data_paths = {
        "tabular1": "path/to/tabular1_data.csv",
        "tabular2": "path/to/tabular2_data.csv",
        "image": "path/to/image_data.pt",
    }

    external_test_data_paths = {
        "tabular1": "path/to/tabular1_test_data.csv",
        "tabular2": "path/to/tabular2_test_data.csv",
        "image": "path/to/image_test_data.pt",
    }

    # Using the training data
    data_module = prepare_fusion_data(prediction_task=...,
                                      fusion_model=some_example_model,
                                      data_paths=data_paths,
                                      output_paths=...)

    # Train the model on the training data
    trained_model= train_and_save_models(data_module=data_module,
                                        fusion_model=some_example_model)

    # Evaluate the model on the external test data
    RealsVsPreds.from_new_data(trained_model, output_paths=..., test_data_paths=external_test_data_paths)

