Quick-Start Script
==================================

This script provides a simple setup to train a model using ``fusilli`` on a single dataset with default parameters.

.. note::

    For a more detailed guide on using Fusilli, refer to the :ref:`train_test_examples`.

This code showcases the necessary steps to execute Fusilli on a single dataset.


Usage Example
-------------

Ensure the elements in the ``params`` dictionary contain specific keys; you can modify the values to match your requirements.


.. code-block:: python


    from fusilli.data import get_data_module
    from fusilli.train import train_and_save_models
    from fusilli.eval import RealsVsPreds
    import matplotlib.pyplot as plt

    # Import the example fusion model
    from fusilli.fusionmodels.tabularfusion.example_model import ExampleModel

    # Set paths to the data
    tabular_1_path = "path/to/tabular_1.csv"
    tabular_2_path = "path/to/tabular_2.csv"
    image_path = "path/to/image_file.pt"

    params = {
        "kfold_flag": False, # Train/test set split
        "log": False, # Use CSV to log losses
        "pred_type": "regression",  # Type: regression, binary, or multiclass classification
        "checkpoint_dir": "path/to/checkpoint/dir",  # Unique dir for each experiment
        "loss_log_dir": "path/to/loss/log/dir",  # Unique dir for each experiment
        "loss_fig_path": "path/to/loss/fig",  # Unique dir for each experiment
        "tabular1_source": tabular_1_path,  # Path to tabular dataset 1
        "tabular2_source": tabular_2_path,  # Path to tabular dataset 2
        "img_source": image_path,  # Path to image dataset
    }

    # Get the data module (PyTorch Lightning-compatible data structure)
    data_module = get_data_module(fusion_model=ExampleModel, params=params)

    # Train the model and receive a list with the trained model
    trained_model_list = train_and_save_models(
        data_module=data_module,
        params=params,
        fusion_model=ExampleModel,
    )

    # Evaluate the model by plotting the real values vs. predicted values
    RealsVsPreds_figure = RealsVsPreds.from_final_val_data(trained_model_list)
    plt.show()


