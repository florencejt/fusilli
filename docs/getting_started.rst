How to quick-run Fusilli
==================================

This script will train a model on a single dataset, using the default parameters.

.. note::

    For a **much** more detailed look at how to use Fusilli, see the :ref:`example_notebooks`.

This code shows everything that you need to run Fusilli on a single dataset.

The elements in the ``params`` dictionary must be there with those specific keys, but you can change the values to suit your needs.

.. code-block:: python


    from fusilli.data import get_data_module
    from fusilli.train import train_and_save_models
    from fusilli.eval import RealsVsPreds

    # import the example fusion model
    from fusilli.fusionmodels.tabularfusion.example_model import ExampleModel

    # set paths to your tabular datasets (CSV files with study_id and pred_label columns)
    tabular_1_path = "path/to/tabular_1.csv"
    tabular_2_path = "path/to/tabular_2.csv"
    image_path = "path/to/image_file.pt"

    params = {
        "kfold_flag": False, # train/test set split
        "log": False, # using CSV to log losses
        "pred_type": "regression", # regression, binary, or multiclass classification
        "checkpoint_dir": "path/to/checkpoint/dir", # needs unique dir for each experiment
        "loss_log_dir": "path/to/loss/log/dir", # needs unique dir for each experiment
        "loss_fig_path": "path/to/loss/fig", # needs unique dir for each experiment
        "tabular1_source": tabular_1_path, # path to tabular dataset 1
        "tabular2_source": tabular_2_path, # path to tabular dataset 2
        "img_source": image_path, # path to image dataset
    }

    # get the data module (PyTorch Lightning-compatible data structure)
    data_module = get_data_module(fusion_model=ExampleModel, params=params)

    # train the model to get list of length 1 containing the trained model
    trained_model_list = train_and_save_models(
        data_module=data_module,
        params=params,
        fusion_model=ExampleModel,
    )

    # evaluate the model by plotting the real values vs. predicted values
    RealsVsPreds_figure = RealsVsPreds.from_final_val_data(trained_model_list)
    plt.show()


