Getting Started
============================

This will show you how to set up the Fusilli library and start fusing!

Installation
------------

.. code-block:: bash

   pip install fusilli


File structure needed
------------------------

- loss_log_dir, loss_fig_path if using CSVLogger
- local_figures
   - Only really needed if using CSVLogger, otherwise you can plt.show() the figures
- checkpoints: need to specify in params dict with key "checkpoint_dir"
   - params["checkpoint_dir"] for loading checkpoints

Example beginning block for getting file structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   params = {}
   params["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S"),

   params["local_fig_path"] = f"local_figures/{params['timestamp']}"
   params["checkpoint_dir"] = f"checkpoints/{params['timestamp']}"
   params["loss_log_dir"] = f"loss_logs/{params['timestamp']}"
   params["loss_fig_path"] = f"{params['local_fig_path']}/losses"

   # make directories
   os.mkdir(params["local_fig_path"]) # local_figures
   os.mkdir(params["checkpoint_dir"]) # checkpoints
   os.mkdir(params["loss_log_dir"]) # loss_log csvs
   os.mkdir(params["loss_fig_path"]) # loss_figures dir within local_figures

   


Simple Example to run fusilli
-----------------------------

.. code-block:: python

   # example of using the library goes here!