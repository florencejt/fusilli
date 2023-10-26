Getting Started
============================

This will show you how to set up the fusilli and get fusing!

Installation
------------

Fusilli is available on PyPI, so you can install it with pip.

.. code-block:: bash

   pip install fusilli


File structure requirements
----------------------------

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

    Fusilli will use pre-determined file names to save the files in these directories. If you have files with the same names in these directories, they will be overwritten.
    It is recommended to have separate directories for each run of fusilli, so that you can keep track of the files that belong to each run.


Example beginning block for getting file structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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




