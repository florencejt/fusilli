Quick-Start Script
==================================

This script provides a simple setup to use ``fusilli`` on a single dataset with default parameters.

.. note::

    For a more detailed guide on using Fusilli, refer to the :ref:`train_test_examples`.

**Before you run this, you need to:**

1. Install ``fusilli`` (see :ref:`install_instructions`).
2. Prepare your data and specify the paths to your data (see :ref:`data-loading`).
3. Specify output file paths (see :ref:`experiment-set-up`).


Running One Model (Regression)
-------------------------------

.. code-block:: python


    from fusilli.data import prepare_fusion_data
    from fusilli.train import train_and_save_models
    from fusilli.eval import RealsVsPreds
    import matplotlib.pyplot as plt

    # Import the example fusion model
    from fusilli.fusionmodels.tabularfusion.concat_data import ConcatTabularData

    data_paths = {
        "tabular1": "path/to/tabular_1.csv",  # Path to tabular dataset 1
        "tabular2": "path/to/tabular_2.csv",  # Path to tabular dataset 2
        "image": "path/to/image_file.pt",  # Path to image dataset
    }

    output_paths = {
        "checkpoints": "path/to/checkpoints/dir",  # Unique dir for each experiment
        "losses": "path/to/losses/dir",  # Unique dir for each experiment
        "figures": "path/to/figures/dir",  # Unique dir for each experiment
    }

    # Get the data module (PyTorch Lightning-compatible data structure)
    data_module = prepare_fusion_data(prediction_task="regression",
                                      fusion_model=ConcatTabularData,
                                      data_paths=data_paths,
                                      output_paths=output_paths)

    # Train the model and receive a list with the trained model
    trained_model = train_and_save_models(data_module=data_module,
                                          fusion_model=ConcatTabularData)

    # Evaluate the model by plotting the real values vs. predicted values
    RealsVsPreds_figure = RealsVsPreds.from_final_val_data(trained_model)
    plt.show()


Comparing Multiple Models (Binary classification)
-------------------------------------------------

Here we show how to compare multiple models on a single dataset.
Instead of using ``plt.show()``, we can also save the figures to a file using ``plt.savefig()``, so let's do that.

.. code-block:: python

    from fusilli.data import prepare_fusion_data
    from fusilli.train import train_and_save_models
    from fusilli.eval import ConfusionMatrix, ModelComparison
    from fusilli.utils.model_chooser import import_chosen_fusion_models
    import matplotlib.pyplot as plt

    data_paths = {
        "tabular1": "path/to/tabular_1.csv",  # Path to tabular dataset 1
        "tabular2": "path/to/tabular_2.csv",  # Path to tabular dataset 2
        "image": "path/to/image_file.pt",  # Path to image dataset
    }

    output_paths = {
        "checkpoints": "path/to/checkpoints/dir",  # Unique dir for each experiment
        "losses": "path/to/losses/dir",  # Unique dir for each experiment
        "figures": "path/to/figures/dir",  # Unique dir for each experiment
    }

    # import a list of models based on conditions: for example, all tabular models
    model_conditions = {
        "modality_type": ["tabular1", "tabular2", "tabular_tabular"],
    }

    fusion_models = import_chosen_fusion_models(model_conditions)

    # Dictionary to store trained models for later comparison
    trained_models = {}

    for model in fusion_models:

        model_name = model.__name__

        # Get the data module (PyTorch Lightning-compatible data structure)
        data_module = prepare_fusion_data(prediction_task="binary",
                                          fusion_model=model,
                                          data_paths=data_paths,
                                          output_paths=output_paths,
                                          )

            # Train the model and receive a list with the trained model
        trained_model = train_and_save_models(data_module=data_module,
                                              fusion_model=model,)

        # Evaluate the model by plotting the confusion matrix
        ConfusionMatrix_figure = ConfusionMatrix.from_final_val_data(trained_model)
        plt.savefig(f"{output_paths['figures']}/{model_name}_confusion_matrix.png")

        # save trained model
        trained_models[model_name] = trained_model

    # compare models
    comparison_fig = ModelComparison.from_final_val_data(trained_models)
    plt.savefig(f"{output_paths['figures']}/model_comparison.png")

