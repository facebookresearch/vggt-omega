# Supervised geometric filtering

This directory provides an optional, user-trained filter for unified-format sequences. Users
need to provide their own binary labels to train the filtering model. `True` means keep a
sequence; the meaning of keep/reject is therefore defined entirely by those labels.

The pipeline computes 38 geometric features and supplies all of them to six tree-based
classifiers. A sequence is rejected when at least two classifiers vote `False`.

## Install

From the repository root:

```bash
pip install -e ".[curation]"
cd training
```

## 1. Compute features

The input is the unified layout documented in the parent [training README](../README.md). Frames
are evaluated at their stored image/depth resolution and in the order listed by
`image_names.json`.

```bash
python -m curation.compute_metrics \
  --data-root /path/to/unified/data \
  --output /path/to/metrics.jsonl
```

Use `--sequence-list` to process a text file containing one sequence name per line. `--mask-path`
mirrors the dataset key:

```bash
python -m curation.compute_metrics \
  --data-root /path/to/unified/data \
  --sequence-list /path/to/sequences.txt \
  --mask-path valid_masks \
  --output /path/to/metrics.jsonl
```

When `--mask-path` is set, every frame must have a readable mask. Extraction failures are recorded
in the metrics cache and cause a nonzero exit. The cache also records its mask setting, seed, and
point cap. Computation is reproducible for the same inputs, software stack, and seed.

## 2. Add your labels

Create a text file with exactly two whitespace-separated fields per line:

```text
scene_a True
scene_b False
```

Blank lines and lines beginning with `#` are ignored. Sequence names must be unique.

## 3. Train

```bash
python -m curation.train \
  --metrics /path/to/metrics.jsonl \
  --labels /path/to/labels.txt \
  --output /path/to/curation_model.joblib
```

Hyperparameters live in [ensemble_config.json](ensemble_config.json). The six models are
CatBoost, Extra Trees, histogram gradient boosting, LightGBM, Random Forest, and XGBoost. The
XGBoost positive-class weight is calculated from the supplied training labels. The command reports
results from a stratified validation split, then fits the saved models on all labeled examples.

The model file uses joblib and can execute code while loading. Load only a model you trained or
otherwise trust. The bundle records the exact feature order, schema version, model names,
configuration, and dependency versions. The loader validates this metadata before prediction.

## 4. Filter sequences

Reuse a metrics cache:

```bash
python -m curation.curate \
  --metrics /path/to/metrics.jsonl \
  --model /path/to/curation_model.joblib \
  --allowlist /path/to/accepted.txt \
  --report /path/to/report.jsonl
```

Or compute from unified data in the same command:

```bash
python -m curation.curate \
  --data-root /path/to/unified/data \
  --model /path/to/curation_model.joblib \
  --allowlist /path/to/accepted.txt \
  --report /path/to/report.jsonl
```

The report records each sequence as `pass`, `reject`, or `error`. The allowlist contains the
`pass` entries, one per line, and can be used as a dataset `sequence_list_file`. `positive_score`
is the uncalibrated mean positive-class score across the six models. Cache-based filtering verifies
that its feature settings match the model, while direct filtering reuses the settings stored in
the model bundle.
