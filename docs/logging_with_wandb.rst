.. _wandb:

Logging with Weights and Biases
====================================================

When running fusilli, setting ``params["log"] = True`` enables logging training and validation behavior to Weights and Biases, facilitated by the ``wandb`` library.

Weights and Biases serves as a free tool for tracking machine learning experiments. To utilise Weights and Biases with Fusilli, creating an account and logging into it is necessary. Follow the instructions `here <https://docs.wandb.ai/quickstart>`_ for account creation.

Details on Fusilli's WandB integration can be found in the function :func:`~fusilli.utils.training_utils.set_logger`. Essentially:

#. When ``params["log"] = True``, Fusilli logs training and validation behavior to Weights and Biases. If ``params["log"] = False``, Fusilli plots loss curves using matplotlib and saves them locally as PNG files.
#. Fusilli creates a project in your WandB account named ``params["project_name"]``. If this project exists, Fusilli uses it; otherwise, it creates a new one. If ``params["project_name"]`` isn't specified, Fusilli generates a project named ``"fusilli"``.
#. Rerunning fusion models with different parameters groups these runs by the fusion model's name.
#. Each run, by default, is tagged with the fusion model's modality type and fusion type. Additional tags can be added via ``extra_log_string_dict`` in :func:`~fusilli.train.train_and_save_models`.
#. With k-fold cross-validation, each fold is logged as a separate run grouped by the fusion model's name and tagged with the current fold number.

For instance, consider specifying ``extra_log_string_dict`` in :func:`~fusilli.train.train_and_save_models` for running :class:`~.EdgeCorrGNN` fusion model with :attr:`~.EdgeCorrGNN.dropout_prob` as 0.2 and logging this to Weights and Biases:

.. note::

    For further information on model modifications in fusilli (e.g., altering dropout probability in :class:`~.EdgeCorrGNN`), refer to :ref:`modifying-models`.

.. code-block:: python

    # Importing data and fusion models etc.

    fusion_model = EdgeCorrGNN

    modification = {
        "EdgeCorrGNN": {
            "dropout_prob": 0.2
        }
    }

    extra_string_for_wandb = {"dropout_prob": 0.2}

    trained_model = train_and_save_models(
        datamodule=datamodule,
        params=params,
        fusion_model=fusion_model,
        extra_log_string_dict=extra_string_for_wandb,
        layer_mods=modification
    )


Upon training and inspecting Weights and Biases, the run will be labeled as ``EdgeCorrGNN_dropout_prob_0.2`` and tagged with ``dropout_prob_0.2``.


**What if you're not using Weights and Biases?**

When not using Weights and Biases, Fusilli plots loss curves and saves them locally as PNG files. In this scenario, the WandB project name is replaced by user-specified tags in the PNG file name. For instance, running the same fusion model without using Weights and Biases will produce a PNG file named ``EdgeCorrGNN_dropout_prob_0.2.png``, saved in ``params["loss_fig_path"]``.