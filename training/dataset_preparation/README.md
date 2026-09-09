# Dataset Preparation

This directory contains instructions for an AI coding agent to prepare a public dataset for
VGGT-Omega. Give the agent the path to the relevant Markdown file and the inputs shown below. You do
not need to paste the file into the conversation.

## Which file should I use?

For a new dataset, run the first two steps in order. The visualization step is optional and can be
used at any time.

1. [download_and_convert.md](download_and_convert.md) downloads the official dataset, converts its
   images, depth maps, and camera metadata to VGGT-Omega's format, and checks that the training code
   can read the result.
2. [clean.md](clean.md) looks for unusable scenes or bad depth pixels. It writes a list of scenes to
   use for training, optional masks that mark invalid depth pixels, a suggested dataset config, and a
   report. It does not modify the converted images, depths, or camera files.
3. [visualize_and_review.md](visualize_and_review.md) creates a local web page for reviewing RGB
   images, depth maps, and 3D reconstructions.

If the dataset has already been converted and checked against VGGT-Omega's format, start with
`clean.md`.

Run the agent from the repository root when possible so that the relative paths in these examples
work. Keep downloaded data, converted datasets, generated masks, and large reports outside the Git
checkout.

## Download and convert

Give the agent [download_and_convert.md](download_and_convert.md) and the official dataset webpage:

```text
Read and follow training/dataset_preparation/download_and_convert.md.

DATASET_WEBPAGE: <paste the official dataset webpage here>

# Optional. Leave blank if these have not been chosen yet.
RAW_DATA_DIR:
UNIFIED_DATASET_DIR:
WORK_DIR:
```

Only `DATASET_WEBPAGE` is required to begin. The optional paths are:

- `RAW_DATA_DIR`: where the unchanged official downloads will be stored.
- `UNIFIED_DATASET_DIR`: where the converted dataset will be stored.
- `WORK_DIR`: where the agent will put scripts, file lists, logs, and reports.

The agent will inspect the official release and license, determine which files are needed, build a
download and conversion process that can resume after interruption, and test the result with the
current VGGT-Omega data loader. It may ask you to accept a license, provide credentials, or choose a
storage location when it cannot do so safely on its own.

`UNIFIED_DATASET_DIR` must be a local directory or a mounted directory that normal Python file APIs
can read.

If the agent is not running from the repository root, give it the absolute path to
`download_and_convert.md`.

## Clean

After the converted dataset has passed the format checks, give the agent [clean.md](clean.md) and the
path to the converted dataset:

```text
Read and follow training/dataset_preparation/clean.md.

UNIFIED_DATASET_DIR: <path to the converted dataset>

# Optional. Leave blank if these have not been chosen or are unavailable.
WORK_DIR:
DATASET_WEBPAGE:
CONVERSION_REPORT:
DATASET_CONFIG:
EVALUATION_DATASET_DIRS:
```

Only `UNIFIED_DATASET_DIR` is required to begin. The optional inputs are:

- `WORK_DIR`: where the agent will put analysis scripts, measurements, review files, and reports.
- `DATASET_WEBPAGE`: the official webpage, used to check how the source data should be interpreted.
- `CONVERSION_REPORT`: notes and test results from the conversion step.
- `DATASET_CONFIG`: an existing VGGT-Omega config to test.
- `EVALUATION_DATASET_DIRS`: evaluation datasets to check whether the same scenes or images also
  appear in the training data.

The agent will check image, depth, and camera quality; compare suspicious samples with ordinary ones;
and decide whether a problem affects the whole dataset, a scene, or only part of a depth map. It will
then write separate cleaning files—such as a list of scenes to keep or masks for bad depth
pixels—and test them with the current VGGT-Omega loader. The converted dataset itself remains
unchanged.

If you do not provide evaluation datasets, the agent will not check whether the training dataset
overlaps with evaluation data.

If the agent is not running from the repository root, give it the absolute path to `clean.md`.

## Visualize and review

Give the agent [visualize_and_review.md](visualize_and_review.md) and either a converted dataset path
or a dataset config:

```text
Read and follow training/dataset_preparation/visualize_and_review.md.

UNIFIED_DATASET_DIR: <path to the converted dataset>

# Optional.
DATASET_CONFIG:
OUTPUT_DIR:
WORK_DIR:
NUM_SAMPLES: 100
SEED: 0
REVIEW_GOAL:
```

The agent will connect [visualize_dataset_template.py](visualize_dataset_template.py) to the real
dataset loader and generate a local review site. The site shows one training sample (a group of
frames) at a time, including an interactive 3D view, RGB images, depth maps, metadata, and review
labels that can be exported as JSON. You do not need to write the dataset adapter yourself.

The `possible_view_discontinuity` label is only a note for human review. Applying that label does not
automatically remove the sample from training.
