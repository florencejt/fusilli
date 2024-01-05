.. _experiment-set-up:

Folder Configuration
===================================================

Fusilli requires specific directories in the local experiment space to operate. These directories are essential for storing checkpoints, loss logs, and evaluation figures.
They are passed into the Fusilli pipeline as elements of a dictionary (similar to data source files - detailed in :ref:`data-loading`).
You can refer to examples in the :ref:`example_notebooks` for usage of this dictionary.

Directory Structure Dictionary
--------------------------------

Here is the dictionary structure for defining necessary directories:


.. code-block:: python

    output_paths = {
        "losses": "{path to save loss logs used for creating loss figures}",
        "figures": "{path to save evaluation figures}",
        "checkpoints": "{path to save checkpoints}",
    }


.. warning::

    Fusilli utilises predetermined file names to save files in these directories. Overwriting may occur if files with the same names exist. **It's highly recommended to maintain separate directories for each Fusilli experiment** to manage files belonging to each run effectively.


Example for Creating Directory Structure
----------------------------------------

Here's an example block to set up the necessary directory structure:

.. code-block:: python

    import os

    # Create a timestamp for the run
    run_name = "Run1"

    # Create directories for this specific run
    os.mkdir(f"local_figures/{run_name}") # local_figures
    os.mkdir(f"checkpoints/{run_name}") # checkpoints
    os.mkdir(f"loss_logs/{run_name}") # loss_log csvs

    # Define paths in the dictionary
    output_paths = {
        "losses": f"loss_logs/{run_name}",
        "figures": f"local_figures/{run_name}",
        "checkpoints": f"checkpoints/{run_name}",
    }


You could also use timestamps to create a unique directory for each run:

.. code-block:: python

    from datetime import datetime

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")


After this and running ``fusilli``, the file structure should look like this:

::

    fusilli_experiment_dir
    ├── fusilli_experiment.py or fusilli_experiment.ipynb
    ├── local_figures
    │   └── Run1
    │       ├── losses
    │       │   └── example_loss_figure.png
    │       └── example_evaluation_figure.png
    ├── checkpoints
    │   └── Run1
    │       └── example_checkpoint.pt
    ├── loss_logs
    │   └── Run1
    │       └── example_loss_log.csv
    └── data
        ├── tabular1.csv
        ├── tabular2.csv
        └── image.pt

