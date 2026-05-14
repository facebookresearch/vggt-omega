# Checkpoint Key Rename Record

This file records code renames that can change `state_dict` keys. Use it when
converting internal training checkpoints into release checkpoints.

The release loader applies these rules with
`rename_state_dict_keys(state_dict, rules)`. Each original key is matched
against the rules once, so overlapping renames such as `token_norm` and
`extra_attention_pre_norm` do not cascade.

## 2026-05-14: Head Naming Pass

Code names:

- `vggt_omega.models.heads.camera_head_linear.CameraHeadLinear`
  -> `vggt_omega.models.heads.camera_head.CameraHead`
- `vggt_omega.models.heads.dpt_linear_head.DPTLinearHead`
  -> `vggt_omega.models.heads.dense_head.DenseHead`

`state_dict` key prefixes:

- `camera_head.` -> `camera_head.`
  - No checkpoint key conversion needed.
- `depth_head.` -> `dense_head.`
  - Checkpoints saved before this pass need this prefix conversion before strict
    loading into the renamed release model.
- `alignment_head.` -> `alignment_head.`
  - No checkpoint key conversion needed.

Prediction keys stay unchanged:

- `pose_enc`
- `depth`
- `depth_conf`
- `alignment_student_embedding`
- `alignment_student_token`

## 2026-05-14: Head Trunk Naming Pass

Code names:

- `extra_attention_blocks` -> `trunk`
- `extra_attention_pre_norm` -> `token_norm`
- `pre_norm` -> `token_norm`
- `extra_attention_in_proj` -> `input_proj`
- `in_proj` -> `input_proj`
- final head output `token_norm` -> `trunk_norm`

`state_dict` key prefixes:

- `camera_head.extra_attention_pre_norm.` -> `camera_head.token_norm.`
- `camera_head.extra_attention_blocks.` -> `camera_head.trunk.`
- `camera_head.token_norm.` -> `camera_head.trunk_norm.`
- `alignment_head.student.pre_norm.` -> `alignment_head.student.token_norm.`
- `alignment_head.student.extra_attention_blocks.` -> `alignment_head.student.trunk.`
- `alignment_head.student.token_norm.` -> `alignment_head.student.trunk_norm.`

The projection modules were renamed in code but do not affect the current
release checkpoints because they are `nn.Identity()` and have no parameters:

- `camera_head.extra_attention_in_proj` -> `camera_head.input_proj`
- `alignment_head.student.in_proj` -> `alignment_head.student.input_proj`
