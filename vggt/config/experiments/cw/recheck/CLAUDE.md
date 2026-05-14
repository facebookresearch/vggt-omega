# Recheck Ablation Notes

This folder contains small-node recheck ablations used to narrow down the
failure cases observed when moving from
`experiments/cw/restart_r256_include_patience_match_t12.yaml` to
`experiments/cw/w003_t4.yaml`.

## Current Intent

- Use small-node experiments to quickly identify which directions are likely
  causing the regression.
- Treat dataset changes as the current top priority.
- Prefer focused directional ablations that are easy to interpret.
- Strict single-variable isolation is not always required once the user wants
  to narrow down a broad direction first.

## Baseline Definitions

- `r001.yaml`
  - Accepted by the user as the baseline for this recheck suite.
  - It should be treated as the default parent for new ablations in this
    folder unless the user explicitly asks otherwise.
  - It is a small-node t12-style control, not a literal full retrain of the
    original `t12` recipe.

- `r002_conf.yaml`
  - Accepted by the user as the confidence-related ablation for this suite.
  - Keep in mind that it is a confidence-related bundle, not a perfectly pure
    conf-only change.

## Non-Data Directions Already Added

- `r002_conf.yaml`
  - Confidence-related bundle.
  - Enables confidence heads/losses and reuses `w003_t4` depth/point head
    initialization.

- `r003.yaml`
  - Point-loss direction.
  - Turns on `point` loss in a way that better mimics `w003_t4` point settings
    while keeping confidence off.

- `r004.yaml`
  - Auxiliary geometry loss bundle.
  - Uses broader `w003_t4`-style depth/preg auxiliary geometry settings while
    keeping confidence off and `point` disabled.

- `r005.yaml`
  - Geometry augmentation bundle.
  - Uses the broader `w003_t4`-style geometry augmentation settings on top of
    `r001`.

- `r006.yaml`
  - Resolution direction.
  - Changes to `img_size: 512` and also matches `w003_t4`'s
    `dyna_ratio: 0.5` to avoid OOM.

- `r023_conf_point_512_no_preg.yaml`
  - Combined non-data bundle.
  - Uses `512` resolution, enables confidence, turns on `point` loss,
    and disables `preg`.

## Dataset Findings So Far

When ignoring sampling-ratio knobs such as `synt_ratio`, `g_ratio`, and
`len_multiplier`, the dataset-side shift from `t12` to `w003_t4` has three
main parts:

1. The training mix expands from `dataset_mix_restart_v5` to
   `dataset_mix_restart_v7`.
2. Many new real, dynamic, and noisier datasets are introduced.
3. `w003_t4` adds dataset-level supervision changes such as
   `disable_depth_loss: True` for several train datasets.

Validation datasets are effectively unchanged; the main shift is in training
datasets and their supervision semantics.

## Dataset Ablation Rule

For dataset-focused ablations in this folder:

- Start from `r001.yaml`.
- Add only the named dataset or dataset family requested by the user.
- Do not silently add unrelated datasets.
- Adjust `len_dataset_scale` so that the newly added dataset or dataset family
  contributes about 30% of train sampling.
- Validate the final mix with:

```bash
python tests/vggt/inspect_sampling_weights.py \
  projects/vggt/config/experiments/cw/recheck/<config>.yaml
```

- If the user asks for a dataset "series", use the exact interpretation listed
  below unless the user says otherwise.

## Dataset Series Conventions Used Here

- `saibo` means all dataset names containing `saibo` that were selected in this
  work:
  - `collected_saibo`
  - `saibo28`
  - `r2_saibo`
  - `r2_saibo_25`

- `crys*` currently means:
  - `crys`
  - `crys48`

- `r2_crys` is not included in the current `crys*` ablation unless the user
  explicitly asks for it.

- `ue` means:
  - `ue`
  - `ue1`
  - `uedec`

- `hor` means:
  - `hor`
  - `hor02`
  - `hor28`

- `eval_series` means:
  - `spring`
  - `sintel`
  - `nrgbd`
  - `da3_hiroom`
  - `da3_scannetpp`
  - `da3_sevenscenes`
  - `dycheck`
  - `uyn`
  - `tum_dynamic`
  - `wai_eth3d_wai`

- `series_a` means:
  - `col`
  - `mvs`
  - `ho_cap`
  - `uco3d`

- `series_b` means:
  - `scannet`
  - `scannet_extra`

- `series_c` means:
  - `interiornet`
  - `staticthings3d`
  - `taskonomy`
  - `wai_gta_sfm`
  - `tex_v2`

- `series_d` means:
  - `syndrone`
  - `cool`
  - `tartanground`
  - `bus`

- `series_e` means:
  - `gen2pilot`
  - `wai_dynamicreplica`
  - `r2_b1`

- `series_f` means:
  - `r2_behavior1k`
  - `rlbench`

- `rigid_series` means:
  - `rigid_v1_top50`
  - `rigid_v2_uncertain`
  - `rigid_v1_todo`

- `c_series` means:
  - `c1`
  - `c2`

## Dataset Ablations Already Added

- `r007_atom.yaml`
  - Adds only `atom`.
  - `len_dataset_scale` override:
    - `atom: 29.72`

- `r008_saibo.yaml`
  - Adds the selected `saibo` family.
  - `len_dataset_scale` overrides:
    - `collected_saibo: 6.09`
    - `saibo28: 6.09`
    - `r2_saibo: 4.06`
    - `r2_saibo_25: 4.06`

- `r009_crys.yaml`
  - Adds the current `crys*` family.
  - `len_dataset_scale` overrides:
    - `crys: 16.5`
    - `crys48: 8.25`

- `r010_rigid_v2.yaml`
  - Adds only `rigid_v2`.
  - `len_dataset_scale` override:
    - `rigid_v2: 1.181`

- `r011_se_col.yaml`
  - Adds only `se_col`.
  - `len_dataset_scale` override:
    - `se_col: 2.126`

- `r012_ue.yaml`
  - Adds the `ue` series.
  - `len_dataset_scale` overrides:
    - `ue: 3.1105`
    - `ue1: 3.1105`
    - `uedec: 6.221`

- `r013_hor.yaml`
  - Adds the `hor` series.
  - `len_dataset_scale` overrides:
    - `hor: 11.5416`
    - `hor02: 11.5416`
    - `hor28: 11.5416`

- `r014_eval_series.yaml`
  - Adds the `eval_series` group.
  - `len_dataset_scale` overrides:
    - `spring: 12.8`
    - `sintel: 4.2`
    - `nrgbd: 7.6`
    - `da3_hiroom: 7.7`
    - `da3_scannetpp: 7.6`
    - `da3_sevenscenes: 4.2`
    - `dycheck: 8.5`
    - `uyn: 4.2`
    - `tum_dynamic: 8.5`
    - `wai_eth3d_wai: 7.7`

- `r015_series_a.yaml`
  - Adds `series_a`.
  - `len_dataset_scale` overrides:
    - `col: 2.40003`
    - `mvs: 0.480006`
    - `ho_cap: 0.26667`
    - `uco3d: 1.33335`

- `r016_series_b.yaml`
  - Adds `series_b`.
  - `len_dataset_scale` overrides:
    - `scannet: 2.78`
    - `scannet_extra: 2.78`

- `r017_series_c.yaml`
  - Adds `series_c`.
  - `len_dataset_scale` overrides:
    - `interiornet: 0.86154`
    - `staticthings3d: 0.43077`
    - `taskonomy: 4.3077`
    - `wai_gta_sfm: 6.46155`
    - `tex_v2: 0.5384625`

- `r018_series_d.yaml`
  - Adds `series_d`.
  - `len_dataset_scale` overrides:
    - `syndrone: 1.27345`
    - `cool: 3.183625`
    - `tartanground: 4.7754375`
    - `bus: 2.5469`

- `r019_series_e.yaml`
  - Adds `series_e`.
  - `len_dataset_scale` overrides:
    - `gen2pilot: 7.4`
    - `wai_dynamicreplica: 37.0`
    - `r2_b1: 9.25`

- `r020_series_f.yaml`
  - Adds `series_f`.
  - `len_dataset_scale` overrides:
    - `r2_behavior1k: 12.8206`
    - `rlbench: 6.4103`

- `r021_rigid_series.yaml`
  - Adds `rigid_series`.
  - `len_dataset_scale` overrides:
    - `rigid_v1_top50: 0.70895`
    - `rigid_v2_uncertain: 0.28358`
    - `rigid_v1_todo: 1.4179`

- `r022_c_series.yaml`
  - Adds `c_series`.
  - `len_dataset_scale` overrides:
    - `c1: 23.56`
    - `c2: 23.56`

## Verified Sampling Result

The following dataset ablations were checked with
`tests/vggt/inspect_sampling_weights.py` and each one was verified to give the
newly added dataset or dataset family about `30%` of train sampling:

- `r007_atom.yaml`
- `r008_saibo.yaml`
- `r009_crys.yaml`
- `r010_rigid_v2.yaml`
- `r011_se_col.yaml`
- `r012_ue.yaml`
- `r013_hor.yaml`
- `r014_eval_series.yaml`
- `r015_series_a.yaml`
- `r016_series_b.yaml`
- `r017_series_c.yaml`
- `r018_series_d.yaml`
- `r019_series_e.yaml`
- `r020_series_f.yaml`
- `r021_rigid_series.yaml`
- `r022_c_series.yaml`

The measured added-dataset share for these configs is:

- `r007_atom.yaml`: `30.007%`
- `r008_saibo.yaml`: `30.007%`
- `r009_crys.yaml`: `30.007%`
- `r010_rigid_v2.yaml`: `30.007%`
- `r011_se_col.yaml`: `30.007%`
- `r012_ue.yaml`: `30.007%`
- `r013_hor.yaml`: `30.006%`
- `r014_eval_series.yaml`: `30.007%`
- `r015_series_a.yaml`: `30.006%`
- `r016_series_b.yaml`: `30.007%`
- `r017_series_c.yaml`: `30.006%`
- `r018_series_d.yaml`: `30.007%`
- `r019_series_e.yaml`: `30.007%`
- `r020_series_f.yaml`: `30.007%`
- `r021_rigid_series.yaml`: `30.006%`
- `r022_c_series.yaml`: `29.983%`

## Current Dataset Coverage Status

For active train datasets, the recheck folder now covers all dataset-side
additions made by `dataset_mix_restart_v7` relative to
`dataset_mix_restart_v5`.

For active train datasets, the recheck folder also covers all dataset-side
additions made by `dataset_mix_restart_v8_t5` relative to
`dataset_mix_restart_v5`.

## Recommended Workflow For Future Additions

1. Start from `r001.yaml` unless the user explicitly wants another parent.
2. Add only the requested dataset or family.
3. Tune `len_dataset_scale` to hit about 30% contribution.
4. Verify with `inspect_sampling_weights.py`.
5. Keep the config name descriptive when the folder grows beyond simple
   numeric naming.

## Open Non-Data Gaps

These non-data directions were identified earlier but do not yet have their
own dedicated recheck config in this folder:

- pairwise camera loss
- RoPE / `force_rope_normalize_coords_max`
- initialization / checkpoint lineage
- optimization recipe

## Important Practical Notes

- Do not "clean up" `r002_conf.yaml` into a purer conf-only ablation unless the
  user explicitly asks. The user already accepted it as satisfying the current
  goal.
- The same principle applies to other recheck configs in this folder: keep the
  existing intent stable unless the user asks for a redesign.
- When adding new dataset ablations, always preserve the current documented
  family interpretation unless the user overrides it.
