Customising Training
=========================================

This page will show you how to customise the training and evaluation of your fusion models.

We will cover the following topics:

* Early stopping
* Valildation metrics
* Batch size
* Number of epochs
* Checkpoint suffix modification
* Number of workers in PyTorch DataLoader
* Train/test and cross-validation splitting yourself

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

Choosing metrics
-----------------

By default, Fusilli uses the following metrics for each prediction task:

* Binary classification: `Area under the ROC curve <https://lightning.ai/docs/torchmetrics/stable/classification/auroc.html>`_ and `accuracy <https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html>`_
* Multiclass classification: `Area under the ROC curve <https://lightning.ai/docs/torchmetrics/stable/classification/auroc.html>`_ and `accuracy <https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html>`_
* Regression: `R2 score <https://lightning.ai/docs/torchmetrics/stable/regression/r2_score.html>`_ and `mean absolute error <https://lightning.ai/docs/torchmetrics/stable/regression/mean_absolute_error.html>`_

You can change the metrics used by passing a list of metrics to the ``metrics_list`` argument in the :func:`~.fusilli.train.train_and_save_models` function.
For example, if you wanted to change the metrics used for a binary classification task to precision, recall, and area under the precision-recall curve, you could do the following:

.. code-block:: python

    new_metrics_list = ["precision", "recall", "auprc"]

    trained_model = train_and_save_models(
        data_module=datamodule,
        fusion_model=example_model,
        metrics_list=new_metrics_list,
        )

Here are the supported metrics as of Fusilli v1.2.0:

**Regression**:

* `R2 score <https://lightning.ai/docs/torchmetrics/stable/regression/r2_score.html>`_: ``r2``
* `Mean absolute error <https://lightning.ai/docs/torchmetrics/stable/regression/mean_absolute_error.html>`_: ``mae``
* `Mean squared error <https://lightning.ai/docs/torchmetrics/stable/regression/mean_squared_error.html>`_: ``mse``

**Binary or multiclass classification**:

* `Area under the ROC curve <https://lightning.ai/docs/torchmetrics/stable/classification/auroc.html>`_: ``auroc``
* `Accuracy <https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html>`_: ``accuracy``
* `Recall <https://lightning.ai/docs/torchmetrics/stable/classification/recall.html>`_: ``recall``
* `Specificity <https://lightning.ai/docs/torchmetrics/stable/classification/specificity.html>`_: ``specificity``
* `Precision <https://lightning.ai/docs/torchmetrics/stable/classification/precision.html>`_: ``precision``
* `F1 score <https://lightning.ai/docs/torchmetrics/stable/classification/f1_score.html>`_: ``f1``
* `Area under the precision-recall curve <https://lightning.ai/docs/torchmetrics/stable/classification/average_precision.html>`_: ``auprc``
* `Balanced accuracy <https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html>`_: ``balanced_accuracy``

If you'd like to add more metrics to fusilli, then please open an issue on the `Fusilli GitHub repository issues page <https://github.com/florencejt/fusilli/issues>`_ or submit a pull request.
The metrics are calculated in :class:`~.fusilli.utils.metrics_utils.MetricsCalculator`, with a separate method for each metric.

**Using your own custom metric:**

If you'd like to use your own custom metric without adding it to fusilli, then you can calculate it using the validation labels and predictions/probabilities.
You can access the validation labels and validation predictions/probabilities from the trained model that is returned by the :func:`~.fusilli.train.train_and_save_models` function.
Look at :class:`~.fusilli.fusionmodels.base_model.BaseModel` for a list of attributes that are available to you to access.


.. note::

    The first metric in the metrics list is used to rank the models in the model comparison evaluation figures.
    Only the first two metrics will be shown in the model comparison figures.
    The rest of the metrics will be shown in the model evaluation dataframe and printed out to the console during training.

.. warning::

    There must be at least two metrics in the metrics list.

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

Checkpoint file names
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


-----

Number of workers in PyTorch DataLoader
-----------------------------------------

You can change the number of workers in the PyTorch DataLoader using the ``num_workers`` argument in the :func:`~.fusilli.data.prepare_fusion_data` function. By default, the number of workers is 0.

.. code-block:: python

    from fusilli.data import prepare_fusion_data
    from fusilli.train import train_and_save_models

    datamodule = prepare_fusion_data(
            prediction_task="binary",
            fusion_model=example_model,
            data_paths=data_paths,
            output_paths=output_path,
            num_workers=4,
        )

    trained_model_list = train_and_save_models(
            data_module=datamodule,
            fusion_model=example_model,
        )



-----

Train/test and cross-validation splitting yourself
---------------------------------------------------

By default, fusilli will split your data into train/test or cross-validation splits for you randomly based on a test size or a number of folds you specify in the :func:`~.fusilli.data.prepare_fusion_data` function.

You can remove the randomness and specify the data indices for train and test, or for the different cross validation folds yourself by passing in optional arguments to :func:`~.fusilli.data.prepare_fusion_data`.


For train/test splitting, the argument `test_indices` should be a list of indices for the test set. To make the test set the first 6 data points in the overall dataset, follow the example below:

.. code-block:: python

    from fusilli.data import prepare_fusion_data
    from fusilli.train import train_and_save_models

    test_indices = [0, 1, 2, 3, 4, 5]

    datamodule = prepare_fusion_data(
            prediction_task="binary",
            fusion_model=example_model,
            data_paths=data_paths,
            output_paths=output_path,
            test_indices=test_indices,
        )

For specifying your own cross validation folds, the argument `own_kfold_indices` should be a list of lists of indices for each fold.

If you wanted to have non-random cross validation folds through your data, you can either specify the folds like so for 3 folds:

.. code-block:: python

    own_kfold_indices = [
        ([ 4,  5,  6,  7,  8,  9, 10, 11], [0, 1, 2, 3]), # first fold
        ([ 0,  1,  2,  3,  8,  9, 10, 11], [4, 5, 6, 7]), # second fold
        ([ 0,  1,  2,  3,  4,  5,  6,  7], [8, 9, 10, 11]) # third fold
    ]

Or to do this automatically, use the Scikit-Learn `KFold functionality <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html>`_ to generate the folds outside of the fusilli functions, like so:

.. code-block:: python

    from sklearn.model_selection import KFold

    num_folds = 5

    own_kfold_indices = [(train_index, test_index) for train_index, test_index in KFold(n_splits=num_folds).split(range(len(dataset)))]


    datamodule = prepare_fusion_data(
        kfold=True,
        prediction_task="binary",
        fusion_model=example_model,
        data_paths=data_paths,
        output_paths=output_path,
        own_kfold_indices=own_kfold_indices,
        num_folds=num_folds,
    )

