# Visualize and Review a VGGT-Omega Dataset

Read inputs from the invoking user message. The user should provide either
`UNIFIED_DATASET_DIR` or `DATASET_CONFIG`; ask one focused question if neither identifies the data.
Optional inputs are `OUTPUT_DIR`, `WORK_DIR`, `NUM_SAMPLES`, `SEED`, and `REVIEW_GOAL`. Resolve paths
named here against the VGGT-Omega repository root.

Build a reproducible visual review from the current public VGGT-Omega data path. Adapt
`training/dataset_preparation/visualize_dataset_template.py` only where dataset-specific sampling is
needed. Run a small end-to-end export when the environment permits instead of stopping after writing
code. Use judgment about scale and review coverage; keep the solution simple.

## Inputs and outputs

- `UNIFIED_DATASET_DIR`: converted dataset root.
- `DATASET_CONFIG`: optional Hydra config or dataset entry whose sampling should be reproduced.
- `OUTPUT_DIR`: generated site and assets; choose a location outside the Git checkout when omitted.
- `WORK_DIR`: adapter, resolved config, manifests, logs, and report, also outside the dataset.
- `NUM_SAMPLES`: default to 100 for the first review.
- `SEED`: choose and record a deterministic seed when omitted.
- `REVIEW_GOAL`: for example, conversion geometry, training sampling, or a cleaning proposal.

Do not put dataset payloads, generated GLBs, or galleries in the repository.

## 1. Define the review

Read `training/data/datasets/unified.py`, `training/data/composed_dataset.py`, the relevant sampler
and config code, and `visual_util.py`. Executable code is authoritative.

Choose and record one data stage:

- converted files, to inspect conversion;
- `UnifiedDataset` output, to include frame selection and spatial/depth preprocessing; or
- `ComposedDataset` output, to inspect tensors and normalization closest to training.

Do not compare normalized values directly with metric source values. Record the code revision,
resolved config, seed, frame counts, aspect ratios, units, and whether sampling represents one
dataset or the complete training mixture.

## 2. Connect the real loader

Copy the Python template into `WORK_DIR`, or import `ReviewSample`,
`review_sample_from_loader`, and `export_review_site` from a small driver. Keep the repository
template dataset-agnostic. Implement `iter_samples` so it yields one unbatched sample at a time:

```python
yield review_sample_from_loader(
    loader_sample,
    sample_id=f"{dataset_name}-{draw_index:05d}",
    frame_labels=selected_frame_ids,
    data_stage="UnifiedDataset-preprocessed",
    metadata={
        "dataset": dataset_name,
        "scene": scene_name,
        "seed": seed,
        "frame_count": requested_frame_count,
        "config_fingerprint": config_fingerprint,
    },
)
```

Use the real loader rather than reimplementing its crop, resize, mask, sampling, or normalization.
Also probe fixed records directly when `__getitem__` retries could hide a failed scene. Seed Python,
NumPy, PyTorch, and the sampler as applicable.

Stable sample IDs identify draws, not just scenes. Preserve the final source frame IDs whenever the
review may support a correction or exclusion. The current `UnifiedDataset` result omits them, and
passing several `ids` fixes only the anchor before the remaining views are selected. Capture final
IDs without changing the sampling result, and verify this with the same seed. If that is not
feasible, label the review exploratory and use view positions rather than invented IDs.

Smoke-test the shared exporter before adapting it:

```bash
pip install -e ".[train,demo]"
python training/dataset_preparation/visualize_dataset_template.py \
  --demo --output /tmp/vggt_omega_review_demo
python -m http.server 8000 --directory /tmp/vggt_omega_review_demo
```

The page uses a pinned public `model-viewer` module from a CDN. In an offline browser environment,
place that module and its license information in the review output and pass its relative path with
`--model-viewer-script`.

## 3. Sample for the question

- Conversion review: cover multiple scenes and source strata, ordinary random records, and known
  edge cases.
- Sampling review: exercise the actual policy across representative frame counts and seeds.
- Cleaning review: include random controls, candidates across the score range, and paired
  before/after samples using identical frames and display scales.
- Training-mixture review: reproduce current mixture and frame-count sampling; do not describe a
  hand-balanced set as representative of training.

Do not select only the first scenes, successful records, or extreme anomalies. Report intended,
exported, and failed counts. Add resumable sharding around the template only when scale requires it.

## 4. Export and review

For each sample, export a merged RGB-coloured point-cloud GLB, all loader-visible RGB frames,
per-frame and sample-shared linear depth colour maps, stable identifiers, the data stage, validity
fractions, and enough config metadata to interpret the draw. Pass loader-provided `point_masks` to
the template: the GLB should show the loader's usable point support, while positive-depth coverage
is reported separately.

Keep binary assets outside `index.html`; the supplied page lazily loads only the selected sample.
Use `--show-cameras` when useful, but compare a point-only view if distant or incorrect cameras
distort framing. Human annotations are stored under a stable run ID and can be exported as JSON;
asset URLs use a separate build revision to avoid stale browser content.

Treat `possible_view_discontinuity` as a review label, not a universal classifier. Dynamic objects,
occlusion, repeated structure, sparse depth, stationary cameras, and valid wide-baseline views can
invalidate simple rigid-scene proxies. A dataset-specific connectivity diagnostic may rank samples
when its meaning is documented, but it must not automatically remove data. Keep connectivity
separate from corrupt geometry, RGB-depth misalignment, empty depth, scale errors, and dynamic
content.

## 5. Validate and report

Check that every manifest entry has its expected assets or an explicit no-geometry state. Reload
GLBs with `trimesh`; verify finite bounds and that point counts match intended loader support after
documented downsampling. Serve the site over HTTP and inspect ordinary and difficult samples. When
browser automation is available, verify rendered pixels, sample switching, frame images, annotation
persistence, and JSON export. Re-run with the same seed and confirm stable sample identities and
ordering.

Report the site path and serve command, adapter and resolved-config paths, intended/exported/failed
counts, represented stage and strata, validation performed, and limitations. State whether
connectivity was reviewed by a person, generated as a diagnostic, or not assessed. Visual review
does not certify the full dataset.
