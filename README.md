<div align="center">

<img src="docs/pink_pasta_logo.png" alt="drawing" width="200"/>

# fusilli 

🌸 **Don't be silly, use fusilli for all your multi-modal data fusion needs!** 🌸

[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white&style=flat)](https://twitter.com/florencetownend)
[![Documentation Status](https://readthedocs.org/projects/fusilli/badge/?version=latest)](https://fusilli.readthedocs.io/en/latest/?badge=latest)
[![version](https://img.shields.io/pypi/v/fusilli)](https://pypi.org/project/fusilli/)
[![downloads](https://img.shields.io/pypi/dm/fusilli)](https://pypi.org/project/fusilli/)

</div>

## Introduction

🍝 Welcome to `fusilli` 🍝, where we're cooking up something special in the world of machine learning! Fusilli is your
go-to
library for multi-modal data fusion methods, and it's designed to make data fusion a piece of cake! 🍰

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

## Detailed Documentation

Want to know more? Here is a link to [Read the Docs](https://fusilli.readthedocs.io/en/latest/)

- Detailed descriptions of the methods included in `fusilli`
- Examples and tutorials on how to use `fusilli`, with examples of:
    - Loading your own data
    - Logging experiments
    - Modifying model structures
- A recipe book of API documentation for the `fusilli` codebase.

## Installation

To savour the flavours of `fusilli`, you can install it using pip:

```
pip install fusilli
```

## How to Cite

Coming soon...

## Contribute!

We'd love to add some more multi-modal data fusion methods if you've found any! PyTorch templates and contribution guidance our in the [contributions documentation](https://fusilli.readthedocs.io/en/latest/contributing_examples/).

## Authors and Acknowledgements

`fusilli` is authored by Florence J Townend, James Chapman, and James H Cole.

This work was funded by the EPSRC (Funding code to be added).

