.. _data-loading:

Loading your Data
==================

``fusilli`` facilitates fusion of **tabular data with tabular data** or **tabular data with images**.

Data Format Requirements
----------------------------

Your data must adhere to specific formats for ``fusilli`` to read it correctly with the :func:`fusilli.data.get_data_module` function.


The paths to the data source files must be in a parameters dictionary:

.. code-block:: python

    params = {
        "tabular1_source": "path/to/tabular1_data.csv",
        "tabular2_source": "path/to/tabular2_data.csv",
        "img_source": "path/to/image_data.pt",
    }

.. warning::

    If you are not using a particular data source, set the value to ``""``.

    For example, if you are not using ``tabular2``, set ``tabular2_source`` in the dictionary to ``""``.

Tabular and Tabular Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

All tabular data must be in CSV format.

Columns named ``study_id`` and ``pred_label`` are required:

- ``study_id``: Unique identifiers for each row.
- ``pred_label``: Labels (integers for classification or floats for regression).


**Example of loading two tabular modalities:**

.. code-block:: python

    from fusilli.data import get_data_module

    params = {
        "tabular1_source": "path/to/tabular1_data.csv",
        "tabular2_source": "path/to/tabular2_data.csv",
        "img_source": "",
    }

    data_module = get_data_module(some_example_model, params)

Tabular and Image Data
~~~~~~~~~~~~~~~~~~~~~~~

Tabular data should follow the format specified above. Image data should be in a ``.pt`` file format with dimensions
``(num_samples, num_channels, height, width)``.

For example, for 100 2D 28x28 grey-scale images, my images.pt file would have the dimensions ``(100, 1, 28, 28)`` when I use ``torch.load()``.

For 100 3D 32x32x32 RGB images, my images.pt file would have the dimensions ``(100, 3, 32, 32, 32)`` when I use ``torch.load()``.

**Example of loading tabular and image data:**

.. code-block:: python

    from fusilli.data import get_data_module

    params = {
        "tabular1_source": "path/to/tabular1_data.csv",
        "tabular2_source": "",
        "img_source": "path/to/image_data.pt",
    }

    data_module = get_data_module(some_example_model, params)

Downsampling Images
*********************

To downsample images before model input, use the ``image_downsample_size`` parameter in the :func:`fusilli.data.get_data_module` function.

**Example of downsampling 2D images to 16x16:**

.. code-block:: python


    data_module = get_data_module(some_example_model, params, image_downsample_size=(16, 16))


-----

Incorporating External Test Data
--------------------------------

For evaluating models with external test data:

- Provide paths to test data sources in the ``params`` dictionary and add suffixes to the dictionary keys (default test suffix is "_test").
- Use the same data format as the training data.

Calling the evaluation figures functions with the method ``from_new_data`` will evaluate the model on the external test data and plot the results.

If you use a different suffix than the default "_test", you must pass the suffix to the evaluation function with the ``data_file_suffix`` parameter.


**Example of training and evaluating a model with external test data:**

.. code-block:: python

    from fusilli.data import get_data_module
    from fusilli.train import train_and_save_models
    from fusilli.eval import RealsVsPreds

    params = {
        "tabular1_source": "path/to/tabular1_training_data.csv",
        "tabular2_source": "path/to/tabular2_training_data.csv",
        "img_source": "path/to/image_training_data.pt",
        "tabular1_source_testing": "path/to/tabular1_test_data.csv",
        "tabular2_source_testing": "path/to/tabular1_test_data.csv",
        "img_source_testing": "path/to/image_test_data.pt",
    }

    # Using the training data (params["tabular1_source"], params["tabular2_source"], and params["img_source"])
    data_module = get_data_module(fusion_model=some_example_model, params=params)

    # Train the model on params["tabular1_source"], params["tabular2_source"], and params["img_source"]
    trained_model= train_and_save_models(data_module, params, some_example_model)

    # Evaluate the model on the external test data:
    # params["tabular1_source_testing"], params["tabular2_source_testing"], and params["img_source_testing"]
    RealsVsPreds.from_new_data(trained_model, params, data_file_suffix="_testing")

