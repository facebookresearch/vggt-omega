# Clean a Converted Public Dataset for VGGT-Omega

Read task inputs from the invoking user message. The user must provide
`UNIFIED_DATASET_DIR`, pointing to a dataset already converted to VGGT-Omega's unified format; if
it is missing, ask for it. They may also provide `WORK_DIR`, `DATASET_WEBPAGE`,
`CONVERSION_REPORT`, `DATASET_CONFIG`, and `EVALUATION_DATASET_DIRS`; treat these as optional and
do not require them before beginning the audit. Resolve repository-relative source paths named in
this document against the VGGT-Omega repository root containing this file, not against an arbitrary
shell working directory.

You are responsible for turning the mechanically valid converted dataset into a documented,
reproducible training view: a sequence allowlist, optional depth-validity masks, and a proposed
dataset configuration, each backed by measurements and review. Work end to end when the environment
allows it. Do not stop after producing anomaly scores if you can inspect the candidates, make
defensible decisions, apply them non-destructively, and validate the result through the real loader.

Use judgment and adapt the work to the actual dataset, annotation source, scale, storage, and
execution environment. The evidence and loader-compatibility requirements remain, but expensive
models, exhaustive pixel scans, parallel jobs, and interactive galleries are only appropriate when
they materially improve the decision. Prefer the simplest reliable and reproducible workflow.

## Scope and decision standard

Cleaning is not a second conversion pass. It decides which otherwise readable supervision should
be exposed to training and how. The task includes:

1. Establishing a reproducible baseline inventory and reading any conversion findings.
2. Measuring image, depth, camera, cross-view, temporal, and sampling quality.
3. Ranking suspicious scenes or frames, then checking those candidates against controls.
4. Choosing the least destructive supported intervention.
5. Producing versioned cleaning artifacts and a candidate dataset configuration.
6. Validating the cleaned view through the repository's actual training data path.

Keep four states distinct throughout the work:

- **conversion error**: the unified contract, association, unit, or camera convention is wrong;
- **confirmed cleaning exclusion**: the data is readable but its supervision is demonstrably wrong
  or unusable for the intended task;
- **review candidate**: a proxy or visual check is suspicious but not decisive;
- **valid or benign outlier**: unusual or difficult data whose supervision remains defensible.

Do not hide a conversion error with a quality filter. Return it to the conversion workflow, state
which scenes are affected, and keep it out of a claimed-clean result until it is repaired and
revalidated. Conversely, do not remove valid data merely because it is difficult, has high model
loss, contains uncommon content, or differs from another dataset's distribution.

This task does not by itself optimize mixture weights, match a benchmark distribution, or prove
that a cleaning choice improves a trained model. It may recommend dataset-specific sampling and
supervision settings needed to make the data usable. Keep those recommendations separate from
claims that would require a controlled training ablation.

## Operating rules

- Treat the current repository source as the authority for what cleaning controls actually do.
  Read `training/README.md` and `training/data/datasets/unified.py`, following related sampling or
  preprocessing code only when needed to resolve the effective behavior. If this prompt disagrees
  with executable code, follow the code and report the discrepancy.
- Read the conversion report, mapping note, validation output, and cleaning-findings file when they
  exist. They are evidence and leads, not automatically established facts.
- Preserve `images/`, `depths/`, `image_names.json`, camera arrays, and `ranking.npy`. The default
  cleaning products are sidecar lists, masks, configuration, metrics, and reports. Never delete or
  overwrite the canonical converted data.
- Do not modify the training loader to accommodate one dataset unless the user explicitly expands
  the task. Recommend only controls supported by the current public loader.
- `UNIFIED_DATASET_DIR` must be local or transparently mounted storage readable through ordinary
  Python file APIs. If `WORK_DIR` is absent, choose a clear writable location outside both the
  dataset root and Git checkout when one is obvious; otherwise ask one focused question.
- Separate measured facts, source-documented facts, interpretations, and decisions in reports.
  Every exclusion or mask rule must be traceable to a reason, rule version, and supporting evidence.
- Do not import fixed thresholds from another dataset. Use source semantics, distributions within
  comparable strata, known controls, visual evidence, and downstream behavior to choose thresholds.
- Treat predictions from a monocular-depth model, segmentation model, vision-language model, or
  other learned system as fallible candidate signals. Disagreement with another model is not proof
  that the dataset ground truth is wrong. Record the model and version, assess possible overlap
  between its training data and the audited dataset, and estimate false positives and false
  negatives on a stratified reviewed subset. No detections is not proof that the failure is absent.
- Do not assume that people, motion, sky, reflections, blur, repeated texture, low baseline, or a
  planar scene is intrinsically invalid. Mask or exclude such content only when it violates the
  intended supervision or an explicit dataset policy.
- Keep generated data and large visualizations outside the Git checkout. Small reusable scripts,
  sequence lists, configuration snippets, and reports may be proposed for the repository only when
  their provenance and redistribution terms permit it.

## Phase 1: establish the baseline

Record the dataset identity, release/version, unified root, conversion report and tool version if
available, existing sequence lists and masks, intended split, and exact scene and frame counts.
Build one deterministic, sorted scene manifest. Preserve stable scene and frame identifiers in all
later artifacts.

Confirm that the input passed the mechanical conversion checks. If no trustworthy full-validation
result exists, run the structural validator from the conversion stage, or implement the minimum
equivalent checks before analyzing quality. Report intended, evaluated, skipped, and failed counts;
do not compute a clean rate from only the scenes that happened to load.

Identify meaningful strata before sampling, such as upstream split, capture device, annotation
pipeline, synthetic versus captured data, camera stream, resolution, environment, or sequence
family. A global average can hide a broken subgroup, and a threshold calibrated on one stratum may
be invalid for another.

Also test for faults shared by most or all scenes, such as a consistent RGB-depth offset, depth
scale or sentinel error, pose timestamp lag, calibration bias, or one annotation-pipeline failure.
Within-dataset ranks and outlier thresholds cannot reveal a common bias. Use source semantics, raw
records, or an independent known-good control. If the fault cannot be separated at scene level, do
not manufacture an allowlist; choose a dataset-wide repair, mask, configuration, supervision change,
or exclusion and validate that intervention directly.

If `DATASET_CONFIG` is supplied, resolve its effective values rather than reading isolated YAML
lines. Otherwise start from a neutral, non-training loader configuration and later emit a minimal
proposed config. Verify the current implementation of at least these controls:

- `sequence_list_file` selects scenes; it is an allowlist, not a frame filter. Check whether
  `train_split_ratio` further partitions it before claiming the final scene membership.
- `mask_path` names a per-scene mask directory. Zero mask pixels zero depth. The current loader may
  warn and continue when a mask is missing or resize a mismatched mask, so the cleaning validator
  must enforce completeness and exact shape itself.
- Depth masks, `force_crop`, `max_depth`, percentile limits, mask erosion, and edge filtering act at
  different stages. Verify their effective order in the current loader, label measurements as raw
  or loader-processed, and inspect how zeros affect percentile calculations.
- `sample_by_index`, `ranking.npy`, `frame_window`, interval sampling, jump behavior, revisit policy,
  and `skip_first_ids` jointly determine the actual multi-frame sample. Analyze the resolved
  combination.
- Evaluate masks and thresholds through the complete data-loading path, not only as standalone
  depth statistics, because they can also change usable geometry and sample validity.

Save the exact baseline configuration and code revision used for every loader-derived measurement.

## Phase 2: measure the dataset before changing it

Start with a small representative sample to validate the analysis, then scale each check according
to its cost. Scan cheap metadata and structural statistics exhaustively. For expensive image,
geometry, or learned-model checks, use deterministic coverage across every stratum, expand around
suspicious cases, and clearly label sampled results as sampled.

A sparse scan is candidate discovery, not certification against isolated frame defects. If the
final decision claims that every retained frame is free of such a defect, inspect every retained
frame or provide a justified bound that supports the claim. Use stable identifiers or a stable hash,
not a process-dependent hash, when selecting deterministic samples.

Keep raw-file measurements separate from loader-preprocessed and composed-normalized measurements.
Every table and visualization must state its stage, applicable depth and translation units, exact
configuration, and sampling rule.

Choose applicable measurements from the following groups. Do not implement a metric merely because
it appears in this list; each one must answer a plausible failure hypothesis for this dataset.
Sanity-check custom metrics with an identity or null case and a targeted perturbation when practical,
and report their support and denominators. If a control does not behave as expected, do not use that
metric to make cleaning decisions.

### Images and frame identity

- successful decoding, dimensions, channels, dtype, and orientation;
- intensity and saturation distributions, blank or nearly constant images, severe blur or encoding
  damage, and persistent invalid borders;
- exact and near-duplicate frames, frozen video segments, discontinuities, or mismatched streams;
- whether a suspected image defect is isolated, persistent within a scene, or characteristic of a
  legitimate capture condition.

Photometric difficulty is not label corruption. Treat blur, darkness, glare, motion, or unusual
content as a cleaning reason only when the input is unusable or its annotations cease to describe
the image. A valid raw frame can still become empty or uninformative after crop or resize. Distinguish
that preprocessing failure from a source-data defect before removing a sequence; if a decision
depends on preprocessing, test the relevant supported parameter range rather than one default view.

### Depth and masks

- valid-pixel coverage and spatial distribution, all-zero or nearly empty frames, and connected
  valid regions;
- positive-depth quantiles, repeated sentinel values, clipping spikes, quantization, implausibly
  flat regions, isolated flying pixels, and discontinuity artifacts;
- behavior by image location, depth range, capture stratum, and time, rather than only a global
  mean;
- RGB-depth registration and frame association using applicable edge, warp, or independent source
  evidence, with deliberately shifted or mismatched controls;
- foreground objects visible in RGB but absent from depth or assigned the background surface;
- coverage and selectivity of every source or generated mask, including what kinds of pixels it
  removes and whether it disproportionately erases a valid depth range or semantic category.

Use absolute depth thresholds only when units and source range are established. Do not choose
`max_depth` from normalized viewer values. Per-frame percentile clipping can remove legitimate near
or far structure and must be evaluated on the actual loader implementation before it is proposed.

### Cameras, geometry, and temporal behavior

- finite and proper poses, plausible intrinsics, per-frame calibration changes, duplicated poses,
  discontinuities, and timestamp or ordering anomalies;
- camera-center path length, displacement, rotation, and adjacent-step distributions. For
  non-metric or mixed-scale data, normalize motion by a robust scene-depth or geometry scale;
- cross-view depth consistency, overlap, positive-depth or cheirality rate, occlusion-aware support,
  and their denominators over multiple frame separations;
- view-graph connectivity, triangulation or parallax support, and geometry-shape proxies when they
  test a concrete failure such as a collapsed reconstruction or panorama-like shell.

Before interpreting cross-view disagreement, determine where the rigid-scene assumption is valid.
Independently moving objects violate rigid reprojection and depth-derived track assumptions even when
their per-frame depth is correct. Report static, dynamic, occluded, and unknown support separately
when feasible; disagreement concentrated on motion is not by itself evidence of bad camera poses or
a reason to exclude the whole scene. A moving object may be valid for per-frame depth while invalid
for point or consistency supervision; if the loader cannot express that distinction, report the
trade-off rather than calling the depth itself wrong.

Trajectory statistics and reconstruction-shape scores are candidate miners, not self-validating
labels. A static camera can be correct, a planar scene can be real, and a reconstruction can fail
without having low parallax. A pair with too little common support cannot establish either
consistency or inconsistency. Confirm the actual failure pattern before excluding a scene.

### Real training samples

Exercise the sampling policy over the configured frame-count range, including representative low,
middle, and high frame counts. Across declared seeds, measure repeated-frame rate, baseline and
overlap distributions, disconnected or nearly empty samples, usable cross-frame support when
required, and the fraction of samples with too little valid supervision.

Do not infer a stable scene property from one random bag. Repeat stochastic measurements and report
medians and spread. If within-scene draw variance dominates between-scene variance or rankings are
unstable across repeats, the metric cannot justify a sequence-removal list; address sampling or
training-time robustness instead. When exact frame pairs are required for an analysis, verify that
the evaluation path preserves them rather than resampling around an anchor.

### Redundancy and evaluation overlap

Measure exact duplicates and, where useful, near-duplicate scenes or frames. Keep low diversity or
heavy redundancy distinct from incorrect supervision: it may justify subsampling or lower exposure,
not a claim that the data is corrupt. When metadata permits, estimate unique capture sessions or
physical environments; a large sequence count alone does not establish diversity.

If `EVALUATION_DATASET_DIRS` or evaluation manifests are provided, check official split membership,
shared identifiers, exact file hashes, and carefully validated near-duplicate evidence. Never infer
leakage from visual similarity alone. If no evaluation corpus is supplied, state explicitly that
evaluation overlap was not assessed rather than calling the result leakage-free.

## Phase 3: turn anomaly signals into decisions

Use automated rules first to rank candidates with high recall. Do not turn the first percentile,
composite score, model disagreement, or extreme value into a final exclusion rule.

For each proposed failure family:

1. State the hypothesis and why it would make supervision wrong or unusable.
2. Inspect examples across its score range, not only the most extreme example.
3. Compare candidates with randomly selected and metric-matched controls from the same relevant
   strata. Use the same frames, support, colour scale, and preprocessing for paired comparisons.
4. Check whether an independent signal supports the decision. If a mask was generated using
   cross-view consistency, the same consistency score is not an independent validation of it.
5. Repeat stochastic measurements and report stability. Prefer robust summaries over the maximum
   from a small sample.
6. Estimate the consequence of the rule: retained scenes, frames, valid pixels, capture strata,
   and sampling exposure.
7. Record confirmed, rejected, and unresolved candidates separately, including reasons.

Measure both sides of every gate: how much confirmed bad supervision it removes and how much
known-good support it removes. When analysis is sharded, merge one result per intended scene or frame
before computing global percentiles, rankings, or top-K selections; concatenated shard-local top-K
lists are not a global top-K.

Report rates using every denominator that materially changes the conclusion, such as scenes,
frames, valid pixels, and expected training exposure. Do not treat these as interchangeable.

Produce a deterministic visual-review set when visual evidence is relevant. Include RGB, depth with
a documented shared scale, RGB-depth overlays, single-frame point clouds, merged multi-frame point
clouds with cameras, and before/after variants as applicable. Include ordinary controls and boundary
cases. A single-frame point cloud cannot reveal every multi-frame registration failure, while a
merged cloud can hide which frame caused an artifact; use both when the distinction matters.
The reusable workflow and exporter template in `training/dataset_preparation/visualize_and_review.md`
and `training/dataset_preparation/visualize_dataset_template.py` may be adapted for this review.

An agent with image-viewing capability should inspect the exported evidence. If the environment
cannot support a defensible review, leave a compact review package and keep those cases labelled as
candidates rather than silently promoting them to exclusions.

## Phase 4: choose and produce the cleaning artifacts

Choose the least destructive intervention that matches the demonstrated failure:

| Evidence | Preferred response |
| --- | --- |
| Wrong convention, unit, calibration, or frame association | Fix and rerun conversion; do not clean around it |
| Scene-wide bad pose or reconstruction | Exclude the scene with `sequence_list_file` |
| Localized wrong depth with otherwise valid image and camera | Add a validated binary depth mask |
| Dataset-wide incorrect depth | Repair and reconvert it, or exclude the affected dataset or stratum |
| Source-documented unusable depth range | Consider `max_depth` in raw units |
| Resampling artifacts at depth boundaries | Evaluate `edge_rtol_thres` or mask erosion on paired examples |
| Repeated, disconnected, or uninformative multi-frame bags | Adjust ranking or supported sampling settings |
| Isolated bad frames | Build a versioned derived dataset and reindex all aligned metadata |
| Redundant but valid data | Subsample or adjust exposure separately from quality exclusions |

### Sequence lists

Write a sorted, unique allowlist using exact scene-directory names, with separate files for upstream
splits when applicable. Also write a structured decision ledger containing every intended scene and
exactly one final status, the reasons, metrics, rule version, and review provenance. Prove that the
ledger has no missing, duplicate, or extra scene IDs and that the allowlist is exactly the intended
accepted subset. Do not publish a denylist when the loader expects an allowlist.

Keep review candidates separate from confirmed exclusions. State explicitly whether unresolved
candidates remain in or out of the proposed training view; do not let a missing decision choose the
policy accidentally.

### Depth masks

The loader expects `<scene>/<mask_path>/<image_stem>.png`, with zero meaning invalid and any nonzero
value meaning valid. Use a new, stable mask-directory name and never overwrite source-provided masks.
Before enabling it, verify that every retained frame has exactly one decodable, single-channel mask
at the native depth resolution. A missing mask must fail cleaning validation even though the loader
may continue without it.

Record how each mask was produced, including source fields or model/version, parameters, and rule
semantics. Measure retained-pixel distributions and inspect both removed and retained boundaries.
Do not replace ground-truth depth with a model prediction. If multiple rules are combined, record
the individual masks and the exact Boolean combination used for the final mask.

### Dataset configuration

Produce a minimal configuration snippet using only keys supported by the current code. Include the
resolved `UNIFIED_DIR`, allowlist, mask directory, raw-unit depth limits, and sampling settings that
were actually justified. Do not copy an old config blindly.

When testing interventions, vary one decision at a time where practical. In particular, do not
attribute a result to scene cleaning if the comparison also changes mixture weight, sampling dose,
frame-count distribution, or multiple unrelated loader settings. Report the usable scene, frame,
and pixel counts so later mixture design can account for the changed exposure.

## Phase 5: validate the proposed cleaned view

Evaluate baseline and cleaned variants on the same deterministic records and common valid support.
Report both quality change and retention cost overall and by relevant stratum. A rule that improves
an aggregate only by dropping nearly all difficult data is not automatically a good cleaning rule.
If a metric changes mechanically with sample size or support, compare against a random or placebo
subset with the same retention rate.

Validate every final sequence-list entry and every generated mask. The validator must fail closed:
report intended, evaluated, skipped, and failed scene/frame counts, and treat an unexpected exception
or absent result as failure rather than aggregating only successful cases.

Run the real repository data path in two modes:

1. Read fixed single-frame records with neutral preprocessing and the proposed allowlist and masks.
   Ensure that a failed record is reported rather than silently replaced by another sample.
2. Exercise the same multi-frame sampling and preprocessing used by the proposed training config,
   with declared seeds and representative frame counts and aspect ratios.

Verify shapes, finite values, nonzero valid support, mask application, sequence membership, sampling
behavior, and usable geometry. Save the exact resolved configuration. If feasible, iterate a short
batch through the configured data pipeline to catch integration failures; do not present it as
evidence of final model quality.

Prove that the proposed config actually consumes each intended allowlist, mask, ranking, and sampling
setting rather than inferring that from config text alone. If an artifact or decision changes after
calibration, rerun the decisive checks on the exact final support.

Re-run the strongest independent quality checks after cleaning and inspect a deterministic paired
before/after visual set. Confirm that the rule removes the intended failure without creating a new
alignment, scale, sampling, or coverage problem.

## Phase 6: finish and report

Apply the validated rules to the full intended population when feasible. If a costly scan or manual
decision remains, preserve the completed measurements, exact resumable command, and candidate set;
label the result partial rather than guessing.

Keep a compact, reproducible artifact set in `WORK_DIR` (names may be adapted):

- the cleaning policy and exact commands;
- the immutable scene manifest and resolved baseline/candidate configs;
- per-scene findings in JSONL or another streaming structured format;
- a candidate list and review manifest;
- the final sequence allowlist and complete decision ledger;
- mask-generation code and a mask manifest, when masks are produced;
- a loader integration test and before/after validation report;
- a concise final report.

Large metrics tables, generated masks, model predictions, and galleries belong outside the Git
checkout. A sequence list or report intended for redistribution must not expose private paths or
assets and must be permitted by the dataset's terms.

The final report must include:

- completion status, clearly distinguishing audit, candidate generation, reviewed decisions,
  artifact generation, and full cleaned-view validation;
- dataset version, source and conversion provenance, paths, code revision, and exact configs;
- intended, evaluated, failed, candidate, accepted, excluded, and unresolved scene/frame counts;
- failure families, evidence, thresholds or model versions, controls, and known limitations;
- before/after quality and retention statistics overall and by important stratum;
- exact paths and hashes for the allowlist, decision ledger, masks, manifests, and reports;
- reproducible commands for sample and full runs;
- whether evaluation overlap was tested and against which exact corpus;
- which conclusions are offline data-quality findings and which, if any, are supported by a
  controlled training experiment.

The work is complete only when every intended scene is accounted for, every enabled cleaning
artifact is complete, the proposed configuration passes the real loader path, and the report's
claims match the strength and coverage of the evidence.
