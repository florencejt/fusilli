---
title: 'Fusilli: A Python package for multimodal data fusion'
tags:
  - Python
  - multimodal data fusion
  - deep learning
  - multiview learning
authors:
  - name: Florence J Townend
    orcid: 0000-0001-7803-5682
    corresponding: true
    affiliation: 1 
  - name: James Chapman
    orcid: 0000-0002-9364-8118
    corresponding: false 
    affiliation: 1
  - name: James H Cole
    orcid: 0000-0003-1908-5588
    corresponding: false 
    affiliation: "1, 2"
affiliations:
 - name: Centre for Medical Image Computing, University College London, UK
   index: 1
 - name: Dementia Research Centre, Institute of Neurology, University College London, UK
   index: 2



date: 12 January 2024
bibliography: paper.bib
---


# Summary

Multimodal data fusion is the integration of data from diverse sources, such as MRI scans, genetics, and clinical measures, to enable predictive analysis that leverages relevant information from all available data modalities.
The terminology used to describe this approach varies widely; multimodal data fusion is also referred to as multi-view, cross-heterogeneous, and multi-source, among others. This nominative inconsistency makes it difficult for people to navigate current research and locate specific data fusion models. 
Moreover, many data-fusion models are underpinned by vastly different architectures, such as graph neural networks, autoencoders, and attention mechanisms.
It remains unclear how to determine the most effective fusion model for a given analysis. Although previous research may indicate the superiority of one over another, comparisons are often made under different conditions. 
Crucially, the level of model complexity needed to optimise the information combination between modalities is unknown. It would be valuable to know the trade-off between model complexity and performance.

To address these issues, `fusilli` allows users to "fuse easily".
It simplifies the comparison of various multimodal data fusion models in predictive tasks. 
Offering a collection of models designed for tabular-tabular and tabular-image fusion, `fusilli` operates as a comprehensive pipeline for training and assessing models across binary or multi-class classification, or regression tasks. 
Its user-friendly interface allows users to modify model structures to suit their specific requirements, empowering them to conduct direct comparisons within their unique settings.

# Statement of need

Multimodal data fusion is applicable to any domain where multiple data modalities are collected. 
Its usage in healthcare and medical domains has increased notably since 2018 [@klineMultimodalMachineLearning2022], owing to the multifactorial nature of medical conditions and the diverse means of assessing the human body. 
These medical domains include but are not limited to oncology [@lipkovaArtificialIntelligenceMultimodal2022], dermatology [@luoArtificialIntelligenceassistedDermatology2023], and neurodegenerative disorders [@huangMultimodalLearningClinically2023].

Data fusion has also been used in an agricultural context to predict crop yield [@s.s.gopiMultimodalMachineLearning2023] or detect diseases [@patilRiceFusionMultimodalityData2022], and in robotics to interpret data from multiple sensors [@duanMultimodalSensorsMLBased2022].
Furthermore, data fusion can be used in analysing disaster response scenarios by integrating information from various sources, including social media posts, images, and audio [@algiriyageMultisourceMultimodalData2021].

Due to the vast array of applications and the relative disconnect between them, there are many distinct machine learning architectures for multimodal data fusion.
Deep learning models in particular are well-suited to multimodal data fusion, as they can learn complex non-linear relationships between modalities.
It is, however, still not clear for researchers to know which models are best for their setting.

To address this, there have been several systematic reviews on the topic of multimodal data fusion [@cuiDeepMultimodalFusion2022; @stahlschmidtMultimodalDeepLearning2022; @gaoSurveyDeepLearning2020; @yanDeepMultiviewLearning2021a]. 
However, these reviews are qualitative, and there is a lack of quantitative benchmarking of models due to non-standardised model implementations.

One solution to this lack of comparability is to create an application-agnostic resource for researchers to be able to easily compare different models in their setting.

Some multimodal data fusion architectures are publicly available (e.g. on GitHub). 
This is useful for researchers who want to use a specific model, but it would be cumbersome for a researcher to exhaustively find and implement all available models for comparison.
Examples of some of these publicly available individual models include `image_tabular` [@tianImage_tabular2020], `MCVAE` [@antelmiSparseMultiChannelVariational2019], and `MADDi` [@golovanevskyMultimodalAttentionbasedDeep2022].

Curated collections offer researchers diverse options for comparison without the need for extensive model sourcing and implementation.
Some collections of multimodal data fusion models focus on non-deep learning models. For instance, `mvlearn` [@perryMvlearnMultiviewMachine2021] is limited to tabular-tabular fusion and focuses on clustering and decomposition rather than deep learning approaches, and `scikit-fusion` [@zitnikScikitfusion2015] (no longer maintained) focuses on latent factor and matrix factorisation models.

As far as we are aware, there are three Python packages with collections of deep learning based multimodal data fusion models: `Multi-view-AE` [@aguilaMultiviewAEPythonPackage2023], `CCA-Zoo` [@chapmanCCAZooCollectionRegularized2021], and `pytorch-widedeep` [@zaurinPytorchwidedeepFlexiblePackage2023].
`Multi-view-AE` is a collection of autoencoder-based models and `CCA-Zoo` is a collection of fusion models based on canonical correlation analysis (CCA). `pytorch-widedeep` is a collection of models based on Google's Wide and Deep algorithm to combine tabular data with either text or images. 

For all three of these packages, the user is required to write their own script for training and evaluation, increasing the time, effort, and expertise needed to run experiments.
However, `fusilli`'s pipeline is readily employable.
Users can complete training and evaluation with just three function calls, while still having the option to extensively customise their experiment.

None of the current packages include models based on graph neural networks or attention mechanisms.
`fusilli` has multiple variations of both of these models and more, covering a wide range of architectures and fusion types.

Additionally, unlike the other data fusion libraries, `fusilli` simplifies model comparison through built-in visualisation methods.
It takes only one line of code to generate a clear figure showing model performances ranked based on the user's chosen performance metric, calculated from either validation or external test data.

Overall, `fusilli` differs from the existing fusion toolkits by providing a comprehensive and flexible pipeline for training, evaluating, and comparing state-of-the-art multimodal data fusion models.


# Implementation

There are four main steps in the `fusilli` pipeline: experiment setup, data preparation, model training, and evaluation and comparison.

### 1. Experiment setup

* Choose the prediction task (binary, multi-class, or regression).
* Import the models to be trained.
* Choose whether to do train/test splitting or k-fold cross-validation.
* Define any model structure modifications.
* Specify experimental parameters, such as early stopping, batch sizes, how to log training, and input data file paths.

### 2. Preparation of data

* Call `prepare_fusion_data` to obtain a PyTorch data module tailored to the model's format.

### 3. Model training

* Call `train_and_save_models` to train a fusion model based on the experimental setup and prepared PyTorch data module.

### 4. Evaluation and comparison

* Call `RealsVsPreds` or `ConfusionMatrix` to generate evaluation figures for a single model, either from validation data or external test data.
* If multiple models have been trained, call `ModelComparison` to generate validation metrics for each fusion model and a figure comparing the models' performance.

## Fusion models in `fusilli`

The table below shows the current list of models in `fusilli`.
`fusilli` categorises models based on the type of fusion, following the taxonomy developed in [@cuiDeepMultimodalFusion2022].
The models are also categorised by the modalities they fuse: tabular-tabular or tabular-image. Some tabular-tabular models have tabular-image counterparts, where the structure of the model lends itself to both types of fusion.

Most of the models in `fusilli` are inspired by methods found in the literature, and references are provided where this is the case. 
These models have been modified to suit the needs and format of `fusilli`, such as simplifying the model, rewriting in PyTorch, or adjusting the architecture to work with tabular-tabular and tabular-image data.
Additionally, some of the models without references may have been used in literature, but they were not inspired by any specific paper because of their relatively ubiquitous implementation.

Importantly, `fusilli` also includes benchmark unimodal models to help users assess whether multimodal data fusion is beneficial for their task.

| Model name (and reference where applicable)                                             | Fusion Category | Modalities Fused |
|-----------------------------------------------------------------------------------------|-----------------|------------------|
| Tabular1 uni-modal                                                                      | Unimodal        | Tabular Only     |
| Tabular2 uni-modal                                                                      | Unimodal        | Tabular Only     |
| Image unimodal                                                                          | Unimodal        | Image Only       |
| Activation function map fusion [@chenMDFNetApplicationMultimodal2023]                   | Operation       | Tabular-tabular  |
| Activation function and tabular self-attention [@chenMDFNetApplicationMultimodal2023]   | Operation       | Tabular-tabular  |
| Concatenating tabular data                                                              | Operation       | Tabular-tabular  |
| Concatenating tabular feature maps [@gaoReducingUncertaintyCancer2022]                  | Operation       | Tabular-tabular  |
| Tabular decision                                                                        | Operation       | Tabular-tabular  |
| Channel-wise multiplication net (tabular) [@duanmuPredictionPathologicalComplete2020]   | Attention       | Tabular-tabular  |
| Tabular Crossmodal multi-head attention [@golovanevskyMultimodalAttentionbasedDeep2022] | Attention       | Tabular-tabular  |
| Attention-weighted GNN [@bintsiMultimodalBrainAge2023]                                  | Graph           | Tabular-tabular  |
| Edge Correlation GNN                                                                    | Graph           | Tabular-tabular  |
| MCVAE Tabular [@antelmiSparseMultiChannelVariational2019 ]                              | Subspace        | Tabular-tabular  |
| Concatenating tabular data with image feature maps [@liFusingMetadataDermoscopy2020]    | Operation       | Tabular-image    |
| Concatenating tabular and image feature maps [@gaoReducingUncertaintyCancer2022]        | Operation       | Tabular-image    |
| Image decision fusion                                                                   | Operation       | Tabular-image    |
| Channel-wise Image attention [@duanmuPredictionPathologicalComplete2020]                | Attention       | Tabular-image    |
| Crossmodal multi-head attention [@golovanevskyMultimodalAttentionbasedDeep2022]         | Attention       | Tabular-image    |
| Trained Together Latent Image + Tabular Data [@zhaoMultimodalDeepLearning2022]          | Subspace        | Tabular-image    |
| Pretrained Latent Image + Tabular Data [@zhaoMultimodalDeepLearning2022]                | Subspace        | Tabular-image    |
| Denoising tabular autoencoder with image maps [@yanRicherFusionNetwork2021]             | Subspace        | Tabular-image    |


## Documentation

The `fusilli` documentation is hosted on Read the Docs (https://fusilli.readthedocs.io) and includes a guide to all the fusion models, installation instructions, tutorials on running experiments and modifying models, and guidance on contributing models to `fusilli`.

# Future Work

We would like to introduce more models to `fusilli` to broaden the available selection.
Additionally, it would be a step forward to modify select models to be able to handle more than two modalities where feasible.
Another objective is to enable users to input images in their original formats, such as JPGs or NIfTIs.

# Conclusion

`fusilli` is a toolkit to compare diverse multimodal data fusion models for predictive tasks. 
It offers an array of models for tabular-tabular and tabular-image data fusion, operating as an efficient pipeline for training and evaluating models across binary, multi-class, and regression tasks. 
Users benefit from the ease of comparing various models within their settings and have the flexibility to adapt model structures to suit their specific requirements.

# Acknowledgements

We would like to thank Sophie Martin and Ana Lawry Aguila for their advice and support in developing `fusilli`, and to Dr Paddy Roddy and Dr Philipp Göbl for their contributions during the 2023 Centre for Medical Image Computing Hackathon.

# References

