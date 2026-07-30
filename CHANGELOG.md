# Changelog

## v2.0.0

Everything below covers all unreleased work since the last published version, v1.2.3.

### Breaking changes

- `data_dims` and data `sources` are now dictionaries (e.g. `{"mod1_dim": ..., "mod2_dim": ...}`) instead of lists,
  to support a variable number of tabular modalities. Anything calling the lower-level classes directly with the
  old list format will need updating; `prepare_fusion_data` users are unaffected.
- Fusion model `forward()` methods now take modalities as separate tensor arguments and return a single tensor,
  instead of taking/returning lists. This makes model outputs compatible with libraries like SHAP that expect a
  single tensor in, single tensor out.
- The binary classification final layer no longer applies `Sigmoid` internally; the loss function now uses
  `BCEWithLogitsLoss` on raw logits directly (more numerically stable). `preds` are still thresholded probabilities
  as before, computed via `sigmoid(logits) > 0.5`.

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
