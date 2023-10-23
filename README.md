# fusilli

### Florence J Townend

Email: [florence.townend.21@ucl.ac.uk](mailto:florence.townend.21@ucl.ac.uk) \
Twitter: [@FlorenceTownend](https://twitter.com/FlorenceTownend)

## Project Description

<img height="150" src="/Users/florencetownend/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Projects/fusilli/docs/pink_pasta_logo.png" width="150"/>

🍝 Welcome to `fusilli` 🍝, where we're cooking up something special in the world of machine learning! Fusilli is your
go-to
library for multi-modal data fusion methods, and it's designed to make data fusion a piece of cake 🍰

*But Florence, what is multi-modal data fusion??* Well, let me tell you...

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

Want to know more? Here is a link to Read the Docs:
[https://fusilli.readthedocs.io/en/latest/](https://fusilli.readthedocs.io/en/latest/)

**In the Read the Docs, you can find:**

- Detailed descriptions of the methods included in `fusilli`
- Examples and tutorials on how to use `fusilli`, with examples of:
    - Loading your own data
    - Logging experiments
    - Modifying model structures
- A recipe book of API documentation for the Fusilli codebase.

### Installation

To savour the flavours of `fusilli`, you can install it using pip:

```
pip install fusilli
```

### Methods

Many of the methods included in `fusilli` are inspired by methods found in literature and have either been adapted from
the original code to fit in with this library's structure or implemented from scratch when no code was available.

The methods are categorised by the modalities they fuse (e.g. tabular-tabular fusion or tabular-image fusion) and by
the type of fusion they perform. These fusion types have been taken from the review by Cui et al (2022) into
data fusion approaches in diagnosis and prognosis [1].

### Authors and Acknowledgements

`fusilli` is authored by Florence J Townend.

This work was funded by the EPSRC (Funding code to be added).

## References

[1] Cui, C., Yang, H., Wang, Y., Zhao, S., Asad, Z., Coburn, L. A., Wilson, K. T., Landman, B. A., & Huo, Y. (2022).
Deep Multi-modal Fusion of Image and Non-image Data in Disease Diagnosis and Prognosis: A Review (arXiv:2203.15588).
arXiv. [https://doi.org/10.48550/arXiv.2203.15588](https://doi.org/10.48550/arXiv.2203.15588)
