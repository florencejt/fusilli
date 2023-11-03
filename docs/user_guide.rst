.. _user-guide:

User guide (installation, data requirements, etc.)
===================================================

Here is the guidance you need to get started with ``fusilli`` before running your experiments: how to install it, how
to input data, and how to choose a model.

How to install
---------------

``fusilli`` is available on PyPI and can be installed with pip:

.. code-block:: bash

    pip install fusilli

You can also install ``fusilli`` from source:

.. code-block:: bash

    git clone ADDRESS GOES HERE
    cd fusilli
    pip install -e .

-----

Data requirements and formats
------------------------------

``fusilli`` can be used to fuse tabular data with tabular, or tabular data with images. Your data must be in the
correct format so that the function :func:`fusilli.data.get_data_module` can read it.

The paths to the data source files must be in the params dictionary with the following keys:

- ``tabular1_source`` : path to the first tabular data source
- ``tabular2_source`` : path to the second tabular data source
- ``img_source`` : path to the image data source

If you are only using models that require 1 tabular data source and 1 image data source, you can omit
``tabular2_source``. Likewise, if you are only using models that require only the tabular data sourcs, you can omit
``img_source``.


Tabular and Tabular
~~~~~~~~~~~~~~~~~~~~

All tabular data must be in csv format. There must be at least two columns with specific names: ``study_id`` and
``pred_label``. The ``study_id`` column must contain the some unique identifiers for each row, and the ``pred_label``
column must contain the labels for each row. The ``study_id`` column will be used to match the rows in the tabular data
with the rows in the image data.

Depending on the task (classification or regression), the ``pred_label`` column must contain different types of data:

- ``Classification``: the ``pred_label`` column must contain integers representing the class labels.
- ``Regression``: the ``pred_label`` column must contain floats representing the predicted values.

**Example of loading two tabular modalities:**

.. code-block:: python

    from fusilli.data import get_data_module
    from fusilli.fusionmodels.tabularfusion import some_example_model # not a real model

    params = {
        "tabular1_source": "path/to/tabular1_data.csv",
        "tabular2_source": "path/to/tabular2_data.csv",
    }

    # get_data_module will use the some_example_model details to know what data to expect.
    # In this case, it will expect tabular1 and tabular2 data.

    data_module = get_data_module(some_example_model, params)

Tabular and Image
~~~~~~~~~~~~~~~~~~~

The tabular data must be in the format described above. The image data must be in the format of a .pt file with the
dimensions (num_samples, num_channels, height, width). The number of samples must be the same as the number of rows in
the tabular data. The number of channels, height, and width can be anything, but they must be the same for all images.

For example, for 100 2D 28x28 grey-scale images, my images.pt file would have the dimensions ``(100, 1, 28, 28)``. For
100 3D 32x32x32 RGB images, my images.pt file would have the dimensions ``(100, 3, 32, 32, 32)``.

If you want to downsample your images when inputting them into the model, you can do so by specifying the
``image_downsample_size`` parameter in the :func:`fusilli.data.get_data_module` function. For example, if you want to
downsample your 2D images to 16x16, you can do so by calling:

.. code-block:: python


    data_module = get_data_module(some_example_model, params, image_downsample_size=(16, 16))

You don't need to specify the number of channels, as the number of channels will be the same as the original image.

**Example of loading tabular and image data**:

.. code-block:: python

    from fusilli.data import get_data_module
    from fusilli.fusionmodels.tabularimagefusion import some_example_model # not a real model

    params = {
        "tabular1_source": "path/to/tabular1_data.csv",
        "img_source": "path/to/image_data.pt",
    }

    # get_data_module will use the some_example_model details to know what data to expect.
    # In this case, it will expect tabular1 and image data.

    data_module = get_data_module(some_example_model, params)


How to incorporate external test data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, models in ``fusilli`` will be trained on the training data and evaluated on the validation data
(including checkpointing with early stopping). However, if you want to evaluate your model on external test data,
you can do so by calling the evaluation figures functions with the method ``from_new_data``. For example, you can
evaluate the model on external test data by calling :func:`fusilli.eval.RealsVsPreds.from_new_data`.

The test data must be in the same format as the training data. The paths to the test data source files must be in the
params dictionary, like the training data. The keys for the test data must have the same names as the training data
but include a suffix to differentiate them. The suffix must be the same for all the test data sources. The suffix
defaults to "_test", but you can change it by passing the ``data_file_suffix`` parameter to the evaluation function.

Some example keys (using ``_testing`` as the suffix) for the test data sources are:

- ``tabular1_source_testing`` : path to the first tabular data source
- ``tabular2_source_testing`` : path to the second tabular data source
- ``img_source_testing`` : path to the image data source

**Example of training a model and evaluating it on external test data:**

.. code-block:: python

    from fusilli.data import get_data_module
    from fusilli.fusionmodels import some_example_model # not a real model
    from fusilli.train import train_and_save_models
    from fusilli.eval import RealsVsPreds

    params = {
        "tabular1_source": "path/to/tabular1_data.csv",
        "tabular2_source": "path/to/tabular2_data.csv",
        "img_source": "path/to/image_data.pt",
        "tabular1_source_testing": "path/to/tabular1_test_data.csv",
        "tabular2_source_testing": "path/to/tabular1_test_data.csv",
        "img_source_testing": "path/to/image_test_data.pt",
    }

    # Get the training data
    data_module = get_data_module(params)

    # Train the model on params["tabular1_source"], params["tabular2_source"], and params["img_source"]
    trained_model_dict = train_and_save_models(data_module, params, some_example_model)

    # Evaluate the model on the external test data:
    # Passing data_file_suffix="_testing" will tell fusilli to look for the test data in the params dictionary with the keys
    # params["tabular1_source_testing"], params["tabular2_source_testing"], and params["img_source_testing"]
    RealsVsPreds.from_new_data(model, params, data_file_suffix="_testing")


-----

How to structure your directories
-----------------------------------

Fusilli needs a few directories to be set up in order to run. This is because fusilli needs somewhere to save the checkpoints, loss logs, and evaluation figures.
The paths to these directories will be passed into the fusilli pipeline as elements of a dictionary.
Examples of using this dictionary are shown in the examples in :ref:`example_notebooks`.

.. code-block:: python

    parameters_dictionary = {
        "loss_fig_path": {path to save loss figures},
        "loss_log_dir": {path to save loss logs, used to make the loss figures},
        "local_fig_path": {path to save evaluation figures},
        "checkpoint_dir": {path to save checkpoints},
    }


.. warning::

    Fusilli uses pre-determined file names to save the files in these directories, such as using the model name to save checkpoints. If you have files with the same names in these directories, **they will be overwritten**.
    It is recommended to have separate directories for each run of fusilli, so that you can keep track of the files that belong to each run.


**Example beginning block for getting file structure:**


.. code-block:: python

    import os
    from datetime import datetime

    # make a timestamp for the run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # make the directories
    os.mkdir(f"local_figures/{timestamp}") # local_figures
    os.mkdir(f"checkpoints/{timestamp}") # checkpoints
    os.mkdir(f"loss_logs/{timestamp}") # loss_log csvs
    os.mkdir(f"local_figures/{timestamp}/losses") # loss_figures dir within local_figures

    # add the paths to the dictionary
    parameters_dictionary = {
        "loss_fig_path": f"local_figures/{timestamp}/losses",
        "loss_log_dir": f"loss_logs/{timestamp}",
        "local_fig_path": f"local_figures/{timestamp}",
        "checkpoint_dir": f"checkpoints/{timestamp}",
    }



-----


How to choose a model
----------------------

``fusilli`` has a number of models that you can use to fuse your data. You can find the list of models in
:mod:`fusilli.fusionmodels`. Each model can be modified to different degrees. More information how to do this is in
:ref:`modifying-models`.

**Ways to choose a model**:

- Choose a model by importing it from :mod:`fusilli.fusionmodels` at the top of your script.
- Use the :func:`fusilli.utils.model_chooser.import_chosen_fusion_models` function to get all the models that fit your criteria.

The input to the :func:`fusilli.utils.model_chooser.import_chosen_fusion_models` function is a dictionary of
criteria. The keys of the dictionary are the names of the criteria, and the values are what you'd like the criteria to
be. For example, if you want to get all the models that can fuse tabular and image data, you can call:

.. code-block:: python

    from fusilli.utils.model_chooser import import_chosen_fusion_models

    criteria = {
        "modality_type": ["tab_img"],
    }

    models = import_chosen_fusion_models(criteria)


:func:`fusilli.utils.model_chooser.import_chosen_fusion_models` will return a list of models that fit the criteria.
You can access the models by indexing the list (``models[0]``, ``models[1]``, etc). The function will also print out a list
of all the models that fit that description.

**More examples of criteria:**

- Models that are using attention-based fusion for tabular and image data:

.. code-block:: python

    criteria = {
        "modality_type": ["tab_img"],
        "fusion_type": ["attention"],
    }

- Models that are using either operation- or attention-based fusion for tabular and image data, and also uni-modal models to benchmark against:

.. code-block:: python

    criteria = {
        "modality_type": ["tab_img", "tabular1", "img"],
        "fusion_type": ["operation", "attention"],
    }

- Models that are any modality type, but are using subspace-based fusion:

.. code-block:: python

    criteria = {
        "fusion_type": ["subspace"],
        "modality_type": "all",
    }

- Getting models by name: Tabular1Unimodal, Tabular2Unimodal, and ConcatTabularData:

.. code-block:: python

    criteria = {
        "class_name": ["Tabular1Unimodal", "Tabular2Unimodal", "ConcatTabularData"],
    }

