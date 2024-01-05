"""
How to customise the training in Fusilli
#########################################

This tutorial will show you how to customise the training of your fusion model.

We will cover the following topics:

* Early stopping
* Batch size
* Number of epochs
* Checkpoint suffix modification

Early stopping
--------------

Early stopping is implemented in Fusilli using the PyTorch Lightning
`EarlyStopping <https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html#lightning.pytorch.callbacks.EarlyStopping>`_
callback. This callback can be passed to the
:func:`~fusilli.model_utils.train_and_save_models` function using the
``early_stopping_callback`` argument. For example:

.. code-block:: python

    from fusilli.data import prepare_fusion_data
    from fusilli.train import train_and_save_models

    from lightning.pytorch.callbacks import EarlyStopping

    modified_early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=3,
        verbose=True,
        mode="min",
    )

    datamodule = prepare_fusion_data(
            prediction_task="binanry",
            fusion_model=example_model,
            data_paths=data_paths,
            output_paths=output_path,
            own_early_stopping_callback=modified_early_stopping_callback,
        )

    trained_model_list = train_and_save_models(
        data_module=datamodule,
        fusion_model=example_model,
        )

Note that you only need to pass the callback to the :func:`~.fusilli.data.prepare_fusion_data` and **not** to the :func:`~.fusilli.train.train_and_save_models` function. The new early stopping measure will be saved within the data module and accessed during training.


-----

Batch size
----------

The batch size can be set using the ``batch_size`` argument in the :func:`~.fusilli.data.prepare_fusion_data` function. By default, the batch size is 8.

.. code-block:: python

    from fusilli.data import prepare_fusion_data
    from fusilli.train import train_and_save_models

    datamodule = prepare_fusion_data(
            prediction_task="binary",
            fusion_model=example_model,
            data_paths=data_paths,
            output_paths=output_path,
            batch_size=32
        )

    trained_model_list = train_and_save_models(
            data_module=datamodule,
            fusion_model=example_model,
            batch_size=32,
        )


-----

Number of epochs
-------------------

You can change the maximum number of epochs using the ``max_epochs`` argument in the :func:`~.fusilli.data.prepare_fusion_data` and :func:`~.fusilli.train.train_and_save_models` functions. By default, the maximum number of epochs is 1000.

You also pass it to the :func:`~.fusilli.data.prepare_fusion_data` function because some of the fusion models require pre-training.

Changing the ``max_epochs`` parameter is especially useful when wanting to run a quick test of your model. For example, you can set ``max_epochs=5`` to run a quick test of your model.

.. code-block:: python

    from fusilli.data import prepare_fusion_data
    from fusilli.train import train_and_save_models

    datamodule = prepare_fusion_data(
            prediction_task="binary",
            fusion_model=example_model,
            data_paths=data_paths,
            output_paths=output_path,
            max_epochs=5,
        )

    trained_model_list = train_and_save_models(
            data_module=datamodule,
            fusion_model=example_model,
            max_epochs=5,
        )

Setting ``max_epochs`` to -1 will train the model until early stopping is triggered.

-----

Checkpoint suffix modification
------------------------------

By default, Fusilli saves the model checkpoints in the following format:

    ``{fusion_model.__name__}_epoch={epoch_n}.ckpt``

If the checkpoint is for a pre-trained model, then the following format is used:

    ``subspace_{fusion_model.__name__}_{pretrained_model.__name__}.ckpt``

You can add suffixes to the checkpoint names by passing a string to the ``extra_log_string_dict`` argument in the :func:`~.fusilli.data.prepare_fusion_data` and :func:`~.fusilli.train.train_and_save_models` functions. For example, I could add a suffix to denote that I've changed the batch size for this particular run:

.. code-block:: python

    from fusilli.data import prepare_fusion_data
    from fusilli.train import train_and_save_models

    extra_suffix_dict = {"batchsize": 32}

    datamodule = prepare_fusion_data(
            prediction_task="binary",
            fusion_model=example_model,
            data_paths=data_paths,
            output_paths=output_path,
            batch_size=32,
            extra_log_string_dict=extra_suffix_dict,
        )

    trained_model_list = train_and_save_models(
            data_module=datamodule,
            fusion_model=example_model,
            batch_size=32,
            extra_log_string_dict=extra_suffix_dict,
        )

The checkpoint name would then be (if the model trained for 100 epochs):

    ``ExampleModel_epoch=100_batchsize_32.ckpt``


.. note::

    The ``extra_log_string_dict`` argument is also used to modify the logging behaviour of the model. For more information, see :ref:`wandb`.
"""
# sphinx_gallery_thumbnail_path = '_static/pink_pasta_logo.png'
