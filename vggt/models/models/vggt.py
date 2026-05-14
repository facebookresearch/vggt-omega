# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import contextlib
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.models.aggregator import Aggregator
from vggt.models.heads.camera_head import CameraHead
from vggt.models.heads.dpt_head import DPTHead
try:
    from vggt.models.heads.text_alignment_head import TextAlignmentHead
except:
    print("TextAlignmentHead is not available")
from vggt.models.heads.track_head import TrackHead
from vggt.models.heads.linear_feat_head import LinearFeatHead

from vggt.models.hub.backbones import dinov3_vit7b16, dinov3_vitb16, dinov3_vitl16, dinov3_vith16plus, dinov3_vits16
from vggt.models.dinov2.hub.backbones import dinov2_vitl14_reg, dinov2_vitb14_reg, dinov2_vits14_reg, dinov2_vitg14_reg

from hydra.utils import instantiate


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, 
                enable_head_amp = False,
                AGGREGATOR = None,
                CAMERA_HEAD = None,
                DEPTH_HEAD = None,
                POINT_HEAD = None,
                RAY_HEAD = None,
                ALIGNMENT_HEAD = None,
                TRACK_HEAD = None,
                MATCH_HEAD = None,
                patch_embed="dinov2_vitl14_reg",
                img_size=512,
                patch_size=16,
                num_register_tokens=4,
                embed_dim=1024,
                init_aggregator_by_patch_embed=False,
                init_aggregator_skip_first_n=0,  # skip the first N layers of patch_embed when initializing aggregator
                init_heads_by_patch_embed=False,
                pos_embed_rope_rescale_coords = None, 
                norm_denominator = None,
                force_rope_normalize_coords_max: bool = False,
                patch_embed_max_depth = None,
                use_checkpoint=True,
                patch_embed_out_indices=None,
                ):
        super().__init__()
        self.patch_embed_out_indices = patch_embed_out_indices
        self.norm_denominator = norm_denominator
        self.force_rope_normalize_coords_max = force_rope_normalize_coords_max
        print(f"VGGT: norm_denominator={self.norm_denominator}")
        print(f"VGGT: force_rope_normalize_coords_max={self.force_rope_normalize_coords_max}")
        self.__build_patch_embed__(
            patch_embed,
            img_size,
            patch_size,
            num_register_tokens,
            embed_dim=embed_dim,
            pos_embed_rope_rescale_coords=pos_embed_rope_rescale_coords,
            norm_denominator=self.norm_denominator,
            force_rope_normalize_coords_max=self.force_rope_normalize_coords_max,
            use_checkpoint=use_checkpoint,
        )

        if TRACK_HEAD is not None:
            raise NotImplementedError("Point and track heads are not supported yet")
            
        self.enable_head_amp = enable_head_amp     


        aggregator_kwargs = {
            "_recursive_": False,
            "use_checkpoint": use_checkpoint,
            "force_rope_normalize_coords_max": self.force_rope_normalize_coords_max,
        }
        if self.force_rope_normalize_coords_max:
            aggregator_kwargs["rope_normalize_coords"] = "max"
        if self.norm_denominator is not None and not self.force_rope_normalize_coords_max:
            aggregator_kwargs["rope_coord_norm_denominator"] = self.norm_denominator
        elif self.norm_denominator is not None and self.force_rope_normalize_coords_max:
            print(
                "VGGT: force_rope_normalize_coords_max=True, ignoring "
                f"aggregator rope_coord_norm_denominator={self.norm_denominator}"
            )
        self.aggregator = instantiate(AGGREGATOR, **aggregator_kwargs)
        self.camera_head = instantiate(CAMERA_HEAD, _recursive_=False) if CAMERA_HEAD is not None else None
        self.depth_head = instantiate(DEPTH_HEAD, _recursive_=False) if DEPTH_HEAD is not None else None                
        self.point_head = instantiate(POINT_HEAD, _recursive_=False) if POINT_HEAD is not None else None
        self.ray_head = instantiate(RAY_HEAD, _recursive_=False) if RAY_HEAD is not None else None
        self.alignment_head = instantiate(ALIGNMENT_HEAD, _recursive_=False) if ALIGNMENT_HEAD is not None else None
        self.match_head = instantiate(MATCH_HEAD, _recursive_=False) if MATCH_HEAD is not None else None
        self.track_head = None
        
        self.init_aggregator_by_patch_embed = init_aggregator_by_patch_embed
        self.init_heads_by_patch_embed = init_heads_by_patch_embed
                
        if self.init_aggregator_by_patch_embed:
            if self.aggregator.use_qk_norm:
                strict_follow_patch_embed = False
            else:   
                strict_follow_patch_embed = True

            with torch.no_grad():
                # Use patch_embed.blocks[skip_first_n:] as source, initialize ALL aggregator blocks
                available_blocks = self.patch_embed.blocks[init_aggregator_skip_first_n:]
                num_available = len(available_blocks)
                num_aggregator_blocks = min(
                    len(self.aggregator.frame_blocks),
                    len(self.aggregator.global_blocks),
                )
                if init_aggregator_skip_first_n > 0:
                    print(f"Skipping first {init_aggregator_skip_first_n} layers of patch_embed for aggregator init")
                    print(f"Using patch_embed.blocks[{init_aggregator_skip_first_n}:] ({num_available} layers) to init {num_aggregator_blocks} aggregator blocks")
                
                for i in range(1, num_aggregator_blocks + 1):
                    # Map from the end: aggregator[-i] <- available_blocks[-i]
                    # If we need more blocks than available, cycle through available blocks again
                    src_idx = -((i - 1) % num_available + 1)  # cycles: -1, -2, ..., -num_available, -1, -2, ...
                    src_state = available_blocks[src_idx].state_dict()
                    self.aggregator.frame_blocks[-i].load_state_dict(src_state, strict=strict_follow_patch_embed)
                    self.aggregator.global_blocks[-i].load_state_dict(src_state, strict=strict_follow_patch_embed)
                del available_blocks
                
        if self.init_heads_by_patch_embed:
            with torch.no_grad():
                available_blocks = self.patch_embed.blocks
                num_available = len(available_blocks)
                
                for head_name, head in [
                    ("camera_head", self.camera_head),
                    ("depth_head", self.depth_head),
                    ("point_head", self.point_head),
                    ("alignment_head", self.alignment_head),
                ]:
                    if head is not None and hasattr(head, "extra_attention_blocks") and head.extra_attention_blocks is not None and len(head.extra_attention_blocks) > 0:
                        num_head_blocks = len(head.extra_attention_blocks)
                        print(f"Initializing {head_name} extra_attention_blocks ({num_head_blocks} layers) from patch_embed (last {num_head_blocks} layers)")
                        
                        # Check strictness based on qk_norm
                        # Assuming head has this attribute if it has extra_attention_blocks
                        head_use_qk_norm = getattr(head, "extra_attention_use_qk_norm", False)
                        if head_use_qk_norm:
                            strict_follow_patch_embed = False
                        else:
                            strict_follow_patch_embed = True
                            
                        for i in range(1, num_head_blocks + 1):
                            # Map from the end: head.extra_attention_blocks[-i] <- available_blocks[-i]
                            src_idx = -i
                            # Handle case where we request more blocks than available
                            if i > num_available:
                                src_idx = -((i - 1) % num_available + 1)
                                
                            src_state = available_blocks[src_idx].state_dict()
                            head.extra_attention_blocks[-i].load_state_dict(src_state, strict=strict_follow_patch_embed)
                
        if patch_embed_max_depth is not None:
            if hasattr(self.patch_embed, "blocks"):
                if len(self.patch_embed.blocks) > patch_embed_max_depth:
                    print(f"Truncating patch_embed from {len(self.patch_embed.blocks)} to {patch_embed_max_depth} layers")
                    self.patch_embed.blocks = self.patch_embed.blocks[:patch_embed_max_depth]
                    if hasattr(self.patch_embed, "n_blocks"):
                        self.patch_embed.n_blocks = patch_embed_max_depth

        # Verify DINO weights are still intact after VGGT init
        if hasattr(self.patch_embed, "patch_embed") and hasattr(self.patch_embed.patch_embed, "proj"):
            proj_weight = self.patch_embed.patch_embed.proj.weight
            print(f"[DINO VERIFY] VGGT.__init__ done - patch_embed.patch_embed.proj.weight: mean={proj_weight.mean().item():.6f}, std={proj_weight.std().item():.6f}")


    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
        pos_embed_rope_rescale_coords = None, 
        norm_denominator = None,
        force_rope_normalize_coords_max: bool = False,
        use_checkpoint=True,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if norm_denominator is not None:
            print(
                "VGGT: forwarding pos_embed_rope_coord_norm_denominator="
                f"{norm_denominator} to patch_embed={patch_embed}"
            )
        if force_rope_normalize_coords_max and norm_denominator is not None:
            print(
                "VGGT: force_rope_normalize_coords_max=True, ignoring "
                f"patch_embed pos_embed_rope_coord_norm_denominator={norm_denominator}"
            )

        effective_norm_denominator = None if force_rope_normalize_coords_max else norm_denominator
        effective_rope_normalize_coords = "max" if force_rope_normalize_coords_max else "separate"

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        elif "dinov3" in patch_embed:
            assert num_register_tokens == 4, "DINO3 only supports 4 register tokens"
            if "vitl16" in patch_embed:
                self.patch_embed = dinov3_vitl16(
                    pretrained=True,
                    pos_embed_rope_rescale_coords=pos_embed_rope_rescale_coords,
                    pos_embed_rope_coord_norm_denominator=effective_norm_denominator,
                    pos_embed_rope_normalize_coords=effective_rope_normalize_coords,
                    use_checkpoint=use_checkpoint,
                )
            elif "vith16plus" in patch_embed:
                self.patch_embed = dinov3_vith16plus(
                    pretrained=True,
                    pos_embed_rope_rescale_coords=pos_embed_rope_rescale_coords,
                    pos_embed_rope_coord_norm_denominator=effective_norm_denominator,
                    pos_embed_rope_normalize_coords=effective_rope_normalize_coords,
                    use_checkpoint=use_checkpoint,
                )
            elif "vitb16" in patch_embed:
                self.patch_embed = dinov3_vitb16(
                    pretrained=True,
                    pos_embed_rope_rescale_coords=pos_embed_rope_rescale_coords,
                    pos_embed_rope_coord_norm_denominator=effective_norm_denominator,
                    pos_embed_rope_normalize_coords=effective_rope_normalize_coords,
                    use_checkpoint=use_checkpoint,
                )
            elif "vits16" in patch_embed:
                self.patch_embed = dinov3_vits16(
                    pretrained=True,
                    pos_embed_rope_rescale_coords=pos_embed_rope_rescale_coords,
                    pos_embed_rope_coord_norm_denominator=effective_norm_denominator,
                    pos_embed_rope_normalize_coords=effective_rope_normalize_coords,
                    use_checkpoint=use_checkpoint,
                )
            elif "vit7b16" in patch_embed:
                self.patch_embed = dinov3_vit7b16(
                    pretrained=True,
                    pos_embed_rope_rescale_coords=pos_embed_rope_rescale_coords,
                    pos_embed_rope_coord_norm_denominator=effective_norm_denominator,
                    pos_embed_rope_normalize_coords=effective_rope_normalize_coords,
                    use_checkpoint=use_checkpoint,
                )
            else:
                raise ValueError(f"Unknown DINO3 backbone: {patch_embed}")
        else:
            dinov2_vit_models = {
                "dinov2_vitl14_reg": dinov2_vitl14_reg,
                "dinov2_vitb14_reg": dinov2_vitb14_reg,
                "dinov2_vits14_reg": dinov2_vits14_reg,
                "dinov2_vitg14_reg": dinov2_vitg14_reg,
            }

            # Hardcode img_size=518 to match DINOv2 pretrained checkpoint size
            # (pos_embed interpolation happens at forward time via interpolate_pos_encoding)
            self.patch_embed = dinov2_vit_models[patch_embed](
                img_size=518,
                patch_size=patch_size,
                block_chunks=block_chunks,
                init_values=init_values,
            )
            # self.patch_embed.patch_embed.proj.weight: mean=0.000005, std=0.013700

        # Disable gradient updates for mask token
        if hasattr(self.patch_embed, "mask_token"):
            self.patch_embed.mask_token.requires_grad_(False)


    def forward(self, batch):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        images = batch["images"]
        
        # print(images.shape)
        if "query_points" in batch:
            query_points = batch["query_points"]
        else:
            query_points = None
            
        # If without batch dimension, add it        
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx, patch_embed_intermediate = self.aggregator(
            images, self.patch_embed, patch_embed_out_indices=self.patch_embed_out_indices
        )

        predictions = {}

        head_amp_context = contextlib.nullcontext() if self.enable_head_amp else torch.cuda.amp.autocast(enabled=False)
        with head_amp_context:
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration, camera from world
                predictions["pose_enc_list"] = pose_enc_list

            if self.depth_head is not None:
                depth, depth_conf, depth_mask = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, patch_embed_intermediate=patch_embed_intermediate
                )
                if depth.shape[-1] == 3:
                    # depth head returns local 3D points [x, y, z]
                    predictions["local_points"] = depth
                    predictions["depth"] = depth[..., 2:3]  # extract z as depth
                else:
                    predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf
                if depth_mask is not None:
                    predictions["depth_mask"] = depth_mask

            if self.point_head is not None:
                pts3d, pts3d_conf, pts3d_mask = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, patch_embed_intermediate=patch_embed_intermediate
                )
                predictions["world_points_reg"] = pts3d
                predictions["world_points_reg_conf"] = pts3d_conf
                if pts3d_mask is not None:
                    predictions["world_points_reg_mask"] = pts3d_mask

            if self.ray_head is not None:
                ray, ray_conf, ray_mask = self.ray_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                    patch_embed_intermediate=patch_embed_intermediate,
                )
                predictions["ray"] = ray
                predictions["ray_conf"] = ray_conf
                if ray_mask is not None:
                    predictions["ray_mask"] = ray_mask

            if self.match_head is not None:
                match_output = self.match_head(
                    aggregated_tokens_list, batch=batch, patch_start_idx=patch_start_idx
                )
                predictions.update(match_output)

            if self.alignment_head is not None:
                alignment_output = self.alignment_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions.update(alignment_output)

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions
