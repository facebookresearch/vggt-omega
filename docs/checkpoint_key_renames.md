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

The projection modules did not affect release checkpoints because they were
`nn.Identity()` and had no parameters. They have since been removed from the
release code:

- `camera_head.extra_attention_in_proj` -> `camera_head.input_proj`
- `alignment_head.student.in_proj` -> `alignment_head.student.input_proj`

## 2026-05-14: Patch Embed Ownership Pass

Code ownership:

- `VGGTOmega.patch_embed` -> `VGGTOmega.aggregator.patch_embed`

The patch embedder now lives inside `Aggregator`, so the top-level model only
wires the aggregator and heads together.

`state_dict` key prefixes:

- `patch_embed.` -> `aggregator.patch_embed.`

## 2026-05-14: Aggregator Naming Pass

Code names:

- `special_global_block_indices` -> `register_attention_block_indices`
- `global_attn_modes` -> `inter_frame_attention_types`
- `_run_global_block` -> `_run_inter_frame_attention_block`
- `patch_start_idx` -> `patch_token_start`
- `special_tokens` -> `camera_and_register_tokens`
- `aggregator.global_blocks` -> `aggregator.inter_frame_blocks`

`state_dict` key prefixes:

- `aggregator.global_blocks.` -> `aggregator.inter_frame_blocks.`

The other renames in this pass are constructor arguments, local variables, or
forward arguments, and do not affect checkpoint keys.

## 2026-05-14: Camera and Text Alignment Naming Pass

Code names:

- `camera_head.pose_branch` -> `camera_head.camera_branch`
- `pose_tokens` -> `camera_tokens`
- `_apply_pose_activation` -> `_apply_camera_activation`
- `pred_pose_enc` -> `raw_camera`
- `VGGTOmega.alignment_head` -> `VGGTOmega.text_alignment_head`
- `SequenceAlignmentStudent` wrapper removed
- `sequence_token` -> `language_token`
- `trunk` -> `readout_blocks`
- final alignment `token_norm` -> `language_token_norm`
- `projector` -> `embedding_projector`

`state_dict` key prefixes:

- `camera_head.pose_branch.` -> `camera_head.camera_branch.`
- `alignment_head.student.pre_norm.` -> `text_alignment_head.token_norm.`
- `alignment_head.student.extra_attention_blocks.` -> `text_alignment_head.readout_blocks.`
- `alignment_head.student.sequence_token` -> `text_alignment_head.language_token`
- `alignment_head.student.token_norm.` -> `text_alignment_head.language_token_norm.`
- `alignment_head.student.projector.` -> `text_alignment_head.embedding_projector.`
- `alignment_head.` -> `text_alignment_head.`

Prediction keys:

- `pose_enc`
- `text_alignment_embedding`
- `text_alignment_token`
