<div align="center">

<img src="docs/_static/pink_pasta_logo.png" alt="drawing" width="200"/>

# fusilli

🌸 **Don't be silly, use fusilli for all your multi-modal data fusion needs!** 🌸

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10137292.svg)](https://doi.org/10.5281/zenodo.10137292)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white&style=flat)](https://twitter.com/florencetownend)
[![Documentation Status](https://readthedocs.org/projects/fusilli/badge/?version=latest)](https://fusilli.readthedocs.io/en/latest/?badge=latest)
[![version](https://img.shields.io/pypi/v/fusilli)](https://pypi.org/project/fusilli/)
[![downloads](https://img.shields.io/pypi/dm/fusilli)](https://pypi.org/project/fusilli/)

</div>

## Introduction

🍝 Welcome to `fusilli` 🍝, the ultimate library for multi-modal data fusion in machine learning! Fusilli makes data
fusion a piece of cake, providing a platform to combine different data types efficiently.

## What can Fusilli do?

Multi-modal data fusion is the combination of different types of data (or data modalities) in the pursuit of some common
goal. For example, using both blood test results and neuroimaging to predict whether somebody will develop a disease.
There are many different ways to combine data modalities, and the aim of `fusilli` is to provide a platform for
anybody to compare different methods against each other.

Fusilli is built using PyTorch Lightning and PyTorch Geometric, and it currently supports the following scenarios:

1. **Tabular-Tabular** **Fusion**: Combine two different types of tabular data.
2. **Tabular-Image** **Fusion**: Combine one type of tabular data with image data (2D or 3D).

Fusilli supports a range of prediction tasks, including **regression**, **binary classification**, and **multi-class
classification.**
Note that it does not currently support tasks such as clustering or segmentation.

Want to know more? Here is a link to [Read the Docs](https://fusilli.readthedocs.io/en/latest/)

## Installation

To savour the flavours of `fusilli`, you can install it using pip:

```
pip install fusilli
```

## Quick Start

Here is a quick example of how to use `fusilli` to train a regression model and plot the real values vs. predicted
values.

```
    from fusilli.data import prepare_fusion_data
    from fusilli.train import train_and_save_models
    from fusilli.eval import RealsVsPreds
    import matplotlib.pyplot as plt

    # Import the example fusion model
    from fusilli.fusionmodels.tabularfusion.example_model import ExampleModel

    data_paths = {
        "tabular1": "path/to/tabular_1.csv",  
        "tabular2": "path/to/tabular_2.csv",  
        "image": "path/to/image_file.pt",  
    }

    output_paths = {
        "checkpoints": "path/to/checkpoints/dir",  
        "losses": "path/to/losses/dir",  
        "figures": "path/to/figures/dir",  
    }

    # Get the data ready
    data_module = prepare_fusion_data(prediction_task="regression",
                                      fusion_model=ExampleModel,
                                      data_paths=data_paths,
                                      output_paths=output_paths)

    # Train the model
    trained_model = train_and_save_models(data_module=data_module,
                                          fusion_model=ExampleModel)

    # Evaluate the model by plotting the real values vs. predicted values
    RealsVsPreds_figure = RealsVsPreds.from_final_val_data(trained_model)
    plt.show()

```

## How to Cite

Florence Townend, Patrick J. Roddy, & Philipp Goebl. (2024). florencejt/fusilli: Fusilli v1.1.0 (v1.1.0).
Zenodo. https://doi.org/10.5281/zenodo.10463697

## Contribute!

If you've developed new fusion methods or want to enhance Fusilli, check our contribution guidelines to get started.
PyTorch templates and contribution guidance our in
the [contributions documentation](https://fusilli.readthedocs.io/en/latest/contributing_examples/).

## Authors and Acknowledgements

`fusilli` is authored by Florence J Townend, James Chapman, and James H Cole.

Florence J Townend is supported by a UCL UKRI Centre for Doctoral Training in AI-enabled Healthcare studentship (
EP/S021612/1).

## License

This project is licensed under AGPLv3. See the LICENSE file for details.
