# Changelog

## v2.0.0

Everything below covers all unreleased work since the last published version, v1.2.3.

### Breaking changes

If you use `prepare_fusion_data` and `train_and_save_models` to run your experiments, most of this doesn't affect
you. The changes below are in the lower-level classes that those functions call internally.

- `data_dims` and data `sources` are now dictionaries (e.g. `{"mod1_dim": ..., "mod2_dim": ...}`) instead of lists,
  to support a variable number of tabular modalities. `prepare_fusion_data`'s own `data_paths` argument was already
  a dictionary before and after this change. This only matters if your code constructs `LoadDatasets`,
  `TrainTestDataModule`, `KFoldDataModule`, or a fusion model class directly with the old list format (for example,
  if you followed the "creating your own fusion model" template) - that code will need updating to the dictionary
  format.
- Fusion model `forward()` methods now take modalities as separate tensor arguments and return a single tensor,
  instead of taking/returning lists. `forward()` is called internally during training and validation, so this only
  matters if you call a model's `.forward()` or `model(x)` directly yourself, e.g. for custom inference or a custom
  fusion model.
- The binary classification final layer no longer applies `Sigmoid` internally, and the loss function now uses
  `BCEWithLogitsLoss` on raw logits directly (more numerically stable). The built-in evaluation classes
  (`RealsVsPreds`, `ConfusionMatrix`, `ModelComparison`) already account for this and don't need any changes. But
  if you use the `logits` returned from a trained model yourself - for your own metrics, manual thresholding, or
  feeding into another library like SHAP - be aware they're no longer bounded between 0 and 1 the way they used to
  be. Thresholding at 0.5 or treating them as probabilities directly will now give wrong results; use
  `sigmoid(logits)` first if you need a probability.

### New features

- **Three-tabular-modality support**: most tabular-tabular fusion methods (`TabularDecision`, `ConcatTabularData`,
  `ConcatTabularFeatureMaps`, `TabularChannelWiseMultiAttention`, `TabularCrossmodalMultiheadAttention`,
  `ActivationFusion`, `AttentionAndSelfActivation`) now accept a third tabular modality via a `tabular3` data source.
  Attention-based methods gained a configurable `main_modality`/`attention_modality` for choosing which modality
  gets special treatment.
- **Image transforms**: `prepare_fusion_data` accepts a `transforms` argument (`torchio.transforms`-style) for
  image augmentation/preprocessing, for 2D and 3D images.
- **GPU/device configuration**: a `training_modifications` argument (accelerator, number of devices) is now
  supported on `prepare_fusion_data` and `train_and_save_models`; metric calculations follow the data's device.
- MCVAE early-stopping patience and tolerance are now configurable via the layer modifications dictionary.

### Robustness fixes

- Fixed the binary classification activation function (see "Breaking changes" above) - models were previously
  double-squashing predictions through a `Sigmoid`.
- Fixed a bug where MCVAE's early-stopping patience attribute was misnamed internally.
- Fixed the custom early-stopping callback.
- Image-based methods no longer error out when the image is too small for the default network architecture -
  they now require `layer_mods` to fix the architecture instead of crashing outright.
- The attention reduction ratio for channel-attention methods now auto-corrects to a working value instead of
  raising when it doesn't evenly divide the modality's feature dimension.
- `torch.load` calls updated for current PyTorch versions (`weights_only`).

### Output / logging improvements

- Training and validation logits are now recorded on trained models, in addition to reals and predictions.
- `ModelComparison` now also returns the underlying reals/preds for further analysis.
- Multiclass dimensions and image downsample size can now be passed to more of the `from_new_data` evaluation
  functions.
- WandB's local log directory is now configurable via an environment variable.
- Loading CSVs with a stray `Unnamed: 0` index column now raises a clear warning instead of silently including it.

### Documentation and packaging

- Documented the three-tabular-modality feature, `main_modality`/`attention_modality` selection, and image
  transforms (previously undocumented).
- Fixed a broken link and a class-name typo in the docs.
- Bumped a vulnerable `urllib3` dependency pin.
- Fixed CI so the vendored `mcvae` git submodule is actually fetched during test runs - this had been silently
  giving the MCVAE fusion model 0% test coverage.
- Fixed a ReadTheDocs build timeout caused by an earlier docs cleanup accidentally discarding the sphinx-gallery
  example cache.
- General branch cleanup and consolidation; test coverage raised from 89% to 93% (304/304 tests passing).
