# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from PIL import Image
from torch.utils.checkpoint import checkpoint

from vggt.models.layers import SelfAttentionBlock

logger = logging.getLogger(__name__)


# Qwen teacher text [0]: Scene: A dimly lit room with a red hue, featuring a door, a window, and a wall with a patterned texture.                                              
# Content: The room contains a closed door on the left, a window on the right, and a wall with a textured pattern. There is a small object on the floor near the window.       
# Appearance: The walls have a floral or damask pattern, the floor is dark, and the overall lighting is red.                                                                   
# Qwen teacher text [1]: Scene: Snowy forest landscape with a winding road.                                                                                                    
# Content: A snow-covered forest with tall pine trees, a winding road, and a steep slope.                                                                                      
# Appearance: The scene has a high-contrast, dramatic look with dark clouds above and bright white snow covering the ground and trees. The trees are covered in snow, and the r
# oad is visible on the slope.                                                                                                                                                 
# Qwen teacher text [2]: Scene: park.                                                                                                                                          
# Content: A tree with a green wrap around its trunk stands in a grassy area, surrounded by bushes and trees.                                                                  
# Appearance: The tree has a thick, cylindrical green wrap around its lower trunk, which is covered in a textured material. The surrounding foliage is lush and green, with som
# e yellowing leaves. The ground is covered in grass and scattered leaves.
# Qwen teacher text [3]: Scene: outdoor public area.
# Content: A dark green plastic picnic table with a hexagonal top and black legs, accompanied by matching benches, situated on a paved surface.
# Appearance: The table and benches are made of a uniform dark green plastic with a matte finish. The structure is supported by black metal legs. The surface of the ground is 
# composed of light gray square tiles.
# Qwen teacher text [4]: Scene: Gallery wall. 
# Content: A gallery wall featuring three framed artworks mounted on a textured, dark wall. The central artwork is a detailed sketch of a figure with a mask-like face. To the 
# left, a smaller framed piece depicts a character in a dynamic pose. To the right, another framed piece shows a floral or abstract pattern. The frames are white and appear to
#  be of a similar size and style.
# Appearance: The wall has a distinct, raised, bumpy texture, possibly resembling a pebble or leaf pattern. The artworks are monochrome, rendered in black and white. The centr
# al piece is the most detailed, while the
# Qwen teacher text [5]: Scene: indoor hallway.
# Content: A wooden floor leads to a doorway with a window, which opens to an outdoor area. The hallway has a wall with a patterned tile design and a wall-mounted light fixtur
# e.
# Appearance: The walls have a textured surface with a geometric pattern. The floor is made of wood planks. The lighting is dim and warm, creating a cozy atmosphere.
# Qwen teacher text [6]: Scene: Urban plaza with modern architecture.
# Content: A central fountain with a large circular sculpture, surrounded by paved walkways and buildings.
# Appearance: The fountain has a metallic, silver-colored structure with a dark base. The surrounding area features light gray stone tiles and modern buildings with glass and 
# concrete facades. The overall design is clean and geometric.


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_ratio: Union[int, float, List[float]],
    *,
    eps: float,
) -> nn.Sequential:
    if isinstance(hidden_ratio, (float, int)):
        hidden_ratio = [hidden_ratio]

    layers = []
    current_dim = input_dim
    for ratio in hidden_ratio:
        hidden_dim = int(input_dim * ratio)
        layers.append(nn.Linear(current_dim, hidden_dim, bias=True))
        layers.append(nn.GELU())
        layers.append(nn.LayerNorm(hidden_dim, eps=eps))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim, bias=True))
    return nn.Sequential(*layers)


def _build_projector(
    input_dim: int,
    output_dim: Optional[int],
    hidden_ratio: Union[int, float, List[float]],
    *,
    eps: float,
    projector_type: str,
) -> Tuple[nn.Module, int]:
    if projector_type == "identity":
        if output_dim is not None and output_dim > 0 and output_dim != input_dim:
            raise ValueError(
                "Identity projector requires output_dim to be unset or match "
                f"the teacher hidden size ({input_dim}), got {output_dim}"
            )
        return nn.Identity(), input_dim

    if projector_type != "mlp":
        raise ValueError(f"Unsupported projector_type: {projector_type}")

    if output_dim is None or output_dim <= 0:
        raise ValueError(
            f"MLP projector requires a positive output_dim, got {output_dim}"
        )

    return (
        _build_mlp(
            input_dim,
            output_dim,
            hidden_ratio,
            eps=eps,
        ),
        output_dim,
    )


class SequenceAlignmentStudent(nn.Module):
    def __init__(
        self,
        dim_in: int = 2048,
        embedding_dim: int = 1024,
        projector_mlp_ratio: Union[int, float, List[float]] = 0.5,
        attention_depth: int = 4,
        attention_dim: int = -1,
        attention_num_heads: int = 16,
        attention_mlp_ratio: float = 4.0,
        attention_qkv_bias: bool = True,
        attention_proj_bias: bool = True,
        attention_ffn_bias: bool = True,
        attention_use_qk_norm: bool = False,
        attention_init_values: float = 1e-5,
        attention_mask_k_bias: bool = True,
        attention_post_norm: bool = True,
        attention_pre_norm: bool = True,
        disable_last_layer_amp: bool = True,
        use_checkpoint: bool = True,
        use_flash_attn: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        if attention_depth <= 0:
            raise ValueError("SequenceAlignmentStudent requires attention_depth > 0")

        attention_dim = dim_in if attention_dim < 0 else attention_dim
        self.disable_last_layer_amp = disable_last_layer_amp
        self.use_checkpoint = use_checkpoint
        self.use_reentrant = False
        self.extra_attention_use_qk_norm = attention_use_qk_norm

        self.pre_norm = (
            nn.LayerNorm(dim_in, eps=eps) if attention_pre_norm else nn.Identity()
        )
        self.in_proj = (
            nn.Identity() if attention_dim == dim_in else nn.Linear(dim_in, attention_dim)
        )
        self.sequence_token = nn.Parameter(torch.zeros(1, 1, attention_dim))
        nn.init.trunc_normal_(self.sequence_token, std=0.02)

        self.extra_attention_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=attention_dim,
                    num_heads=attention_num_heads,
                    ffn_ratio=attention_mlp_ratio,
                    qkv_bias=attention_qkv_bias,
                    proj_bias=attention_proj_bias,
                    ffn_bias=attention_ffn_bias,
                    init_values=attention_init_values,
                    use_qk_norm=attention_use_qk_norm,
                    mask_k_bias=attention_mask_k_bias,
                    use_flash_attn=use_flash_attn,
                )
                for _ in range(attention_depth)
            ]
        )
        self.token_norm = (
            nn.LayerNorm(attention_dim, eps=eps) if attention_post_norm else nn.Identity()
        )
        self.projector = _build_mlp(
            attention_dim,
            embedding_dim,
            projector_mlp_ratio,
            eps=eps,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        patch_start_idx: int,
    ) -> Dict[str, torch.Tensor]:
        if patch_start_idx is None:
            raise ValueError("patch_start_idx is required for alignment student")

        if patch_start_idx > tokens.shape[2]:
            raise ValueError(
                f"patch_start_idx ({patch_start_idx}) exceeds token length ({tokens.shape[2]})"
            )

        if self.disable_last_layer_amp and tokens.dtype != torch.float32:
            tokens = tokens.float()

        batch_size, num_frames, _, _ = tokens.shape
        special_tokens = tokens[:, :, :patch_start_idx, :]
        special_tokens = self.pre_norm(special_tokens)
        special_tokens = self.in_proj(special_tokens)
        special_tokens = special_tokens.reshape(batch_size, num_frames * patch_start_idx, -1)

        sequence_token = self.sequence_token.expand(batch_size, -1, -1)
        joint_tokens = torch.cat([sequence_token, special_tokens], dim=1)
        rope_sincos = None

        for block in self.extra_attention_blocks:
            if self.training and self.use_checkpoint:
                joint_tokens = checkpoint(
                    block,
                    joint_tokens,
                    rope_sincos,
                    use_reentrant=self.use_reentrant,
                )
            else:
                joint_tokens = block(joint_tokens, rope_sincos)

        sequence_token = self.token_norm(joint_tokens[:, 0])
        projected = self.projector(sequence_token)
        return {
            "alignment_student_embedding": F.normalize(projected, dim=-1),
            "alignment_student_token": sequence_token,
        }


class MockSequenceTeacher(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 32,
        normalize: bool = True,
    ):
        super().__init__()
        self.normalize = normalize
        self.output_embedding_dim = embedding_dim
        self.projector = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Cheap deterministic teacher for tests and local smoke checks.
        pooled = images.float().mean(dim=(1, 3, 4))
        embedding = self.projector(pooled)
        if self.normalize:
            embedding = F.normalize(embedding, dim=-1)
        return {
            "alignment_teacher_embedding": embedding,
            "alignment_teacher_text": None,
        }


class QwenVLTeacher(nn.Module):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        prompt: str = (
            "These images show the same scene from multiple viewpoints.\n"
            "Describe the full scene as one coherent scene, not as separate images.\n\n"
            "Use the following format:\n"
            "Scene: <scene category or environment>.\n"
            "Content: <main objects and coarse spatial arrangement>.\n"
            "Appearance: <stable visual attributes such as material, color, or style>.\n\n"
            "Write one short sentence for each field.\n"
            "Mention information that is consistent across multiple views, even if it is not visible in every view.\n"
            "Focus on shared scene content, major objects, coarse layout, and stable appearance.\n"
            "Mention dynamic objects only if they are prominent and identifiable in multiple views.\n"
            "Do not describe motion, actions, or frame-specific changes.\n"
            "Do not describe each image separately.\n"
            "Do not mention camera motion, viewpoint order, image quality, blur, exposure, lighting artifacts, or uncertainty.\n"
            "Keep the description concise, factual, and semantically dense."
        ),
        embedding_dim: int = 1024,
        projector_mlp_ratio: Union[int, float, List[float]] = 0.5,
        projector_type: str = "mlp",
        torch_dtype: str = "bfloat16",
        attn_implementation: Optional[str] = None,
        max_new_tokens: int = 128,
        do_sample: bool = True,
        temperature: float = 0.3,
        top_p: float = 0.85,
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.05,
        trust_remote_code: bool = True,
        compute_on_eval: bool = False,
        log_text: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.model_name = model_name
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.trust_remote_code = trust_remote_code
        self.compute_on_eval = compute_on_eval
        self.log_text = log_text
        self._model = None
        self._processor = None
        self._torch_dtype = self._resolve_dtype(torch_dtype)
        self._attn_implementation = attn_implementation
        self.embedding_dim = embedding_dim
        self.projector_mlp_ratio = projector_mlp_ratio
        self.projector_type = projector_type
        self.projector_eps = eps
        self.projector = None
        self.output_embedding_dim = None
        self._lazy_init()

    @staticmethod
    def _is_primary_rank() -> bool:
        return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

    @staticmethod
    def _resolve_dtype(torch_dtype: str) -> torch.dtype:
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if torch_dtype not in mapping:
            raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")
        return mapping[torch_dtype]

    def _lazy_init(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        try:
            import transformers
        except ImportError as exc:
            raise ImportError(
                "QwenVLTeacher requires `transformers`. Install a recent build "
                "before enabling the alignment branch."
            ) from exc

        processor_cls = getattr(transformers, "AutoProcessor", None)
        if processor_cls is None:
            raise ImportError("transformers.AutoProcessor is required for QwenVLTeacher")
        model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
        if model_cls is None:
            raise ImportError(
                "transformers.AutoModelForImageTextToText is required for QwenVLTeacher"
            )

        model_kwargs = {
            "dtype": self._torch_dtype,
            "trust_remote_code": self.trust_remote_code,
        }
        if self._attn_implementation is not None:
            model_kwargs["attn_implementation"] = self._attn_implementation

        self._model = model_cls.from_pretrained(self.model_name, **model_kwargs)

        self._processor = processor_cls.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        hidden_size = self._get_hidden_size()
        self.projector, self.output_embedding_dim = _build_projector(
            hidden_size,
            self.embedding_dim,
            self.projector_mlp_ratio,
            eps=self.projector_eps,
            projector_type=self.projector_type,
        )
        self._model.requires_grad_(False)
        self._model.eval()

    def _get_hidden_size(self) -> int:
        assert self._model is not None
        config = self._model.config
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is not None:
            return hidden_size

        text_config = getattr(config, "text_config", None)
        if isinstance(text_config, dict):
            hidden_size = text_config.get("hidden_size")
        elif text_config is not None:
            hidden_size = getattr(text_config, "hidden_size", None)

        if hidden_size is None:
            raise RuntimeError(
                f"Unable to infer hidden_size for teacher model {self.model_name}"
            )
        return hidden_size

    def train(self, mode: bool = True):
        super().train(mode)
        if self._model is not None:
            self._model.eval()
        return self

    def _tensor_to_pil(self, frame: torch.Tensor) -> Image.Image:
        frame_uint8 = frame.detach().clamp(0, 1).mul(255).to(torch.uint8)
        frame_uint8 = frame_uint8.permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(frame_uint8)

    def _prepare_messages(self, frames: List[Image.Image]) -> List[Dict]:
        content = [{"type": "image", "image": image} for image in frames]
        content.append({"type": "text", "text": self.prompt})
        return [{"role": "user", "content": content}]

    def _pool_generated_tokens(
        self,
        sequences: torch.Tensor,
        prompt_lengths: torch.Tensor,
        hidden_states: torch.Tensor,
        eos_token_id: Optional[int],
        pad_token_id: Optional[int],
    ) -> torch.Tensor:
        batch_size, seq_len = sequences.shape
        positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0)
        valid_mask = positions >= prompt_lengths.unsqueeze(1)
        if eos_token_id is not None:
            valid_mask &= sequences.ne(eos_token_id)
        if pad_token_id is not None:
            valid_mask &= sequences.ne(pad_token_id)

        counts = valid_mask.sum(dim=1)
        masked_hidden = hidden_states * valid_mask.unsqueeze(-1)
        pooled = masked_hidden.sum(dim=1) / counts.clamp_min(1).unsqueeze(-1)

        fallback_indices = (prompt_lengths - 1).clamp_min(0)
        fallback_hidden = hidden_states[
            torch.arange(batch_size, device=hidden_states.device),
            fallback_indices,
        ]
        has_generated_tokens = counts.gt(0).unsqueeze(-1)
        return torch.where(has_generated_tokens, pooled, fallback_hidden)

    def _move_model_to_device(self, device: torch.device) -> None:
        self._lazy_init()
        assert self._model is not None
        assert self._processor is not None
        assert self.projector is not None

        if next(self._model.parameters()).device != device:
            self._model.to(device)
        projector_param = next(self.projector.parameters(), None)
        if projector_param is not None and projector_param.device != device:
            self.projector.to(device)

    def _prepare_batched_inputs(self, images: torch.Tensor, device: torch.device) -> Dict:
        messages = []
        for sample_images in images:
            frames = [self._tensor_to_pil(frame) for frame in sample_images]
            messages.append(self._prepare_messages(frames))

        try:
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Batched Qwen-VL preprocessing failed. The current processor build does "
                "not appear to support batched multi-image chat-template inputs."
            ) from exc

        if hasattr(inputs, "to"):
            return inputs.to(device)
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

    def _build_generation_kwargs(self) -> Dict:
        assert self._processor is not None
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "return_dict_in_generate": True,
            "pad_token_id": getattr(self._processor.tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(self._processor.tokenizer, "eos_token_id", None),
            "repetition_penalty": self.repetition_penalty,
        }
        generation_kwargs = {
            key: value for key, value in generation_kwargs.items() if value is not None
        }
        if self.do_sample:
            generation_kwargs.update(
                {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                }
            )
            generation_kwargs = {
                key: value for key, value in generation_kwargs.items() if value is not None
            }
        return generation_kwargs

    def _decode_generated_text(
        self,
        prompt_lengths: torch.Tensor,
        sequences: torch.Tensor,
    ) -> List[str]:
        assert self._processor is not None
        generated_ids_trimmed = [
            out_ids[prompt_len:]
            for prompt_len, out_ids in zip(prompt_lengths.tolist(), sequences)
        ]
        return self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def _forward_batched(self, images: torch.Tensor) -> Tuple[torch.Tensor, Optional[List[str]]]:
        device = images.device
        self._move_model_to_device(device)
        inputs = self._prepare_batched_inputs(images, device)
        generation_kwargs = self._build_generation_kwargs()

        prompt_attention_mask = inputs.get("attention_mask")
        if prompt_attention_mask is None:
            prompt_lengths = torch.full(
                (inputs["input_ids"].shape[0],),
                inputs["input_ids"].shape[1],
                device=inputs["input_ids"].device,
                dtype=torch.long,
            )
        else:
            prompt_lengths = prompt_attention_mask.sum(dim=1).to(torch.long)

        with torch.no_grad():
            generated = self._model.generate(**inputs, **generation_kwargs)
            sequences = generated.sequences
            pad_token_id = generation_kwargs.get("pad_token_id")
            if pad_token_id is not None:
                attention_mask = sequences.ne(pad_token_id).long()
            else:
                attention_mask = torch.ones_like(sequences, device=sequences.device)
            forward_inputs = {
                key: value
                for key, value in inputs.items()
                if key not in {"input_ids", "attention_mask"}
            }
            outputs = self._model(
                input_ids=sequences,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                **forward_inputs,
            )
            hidden_states = outputs.hidden_states[-1]
            pooled = self._pool_generated_tokens(
                sequences,
                prompt_lengths,
                hidden_states,
                generation_kwargs.get("eos_token_id"),
                generation_kwargs.get("pad_token_id"),
            )

        caption_text = self._decode_generated_text(prompt_lengths, sequences) if self.log_text else None
        return pooled, caption_text

    def forward(self, images: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        if not self.training and not self.compute_on_eval:
            return {
                "alignment_teacher_embedding": None,
                "alignment_teacher_text": None,
            }

        teacher_hidden, teacher_text = self._forward_batched(images)
        projector_autocast = contextlib.nullcontext()
        if teacher_hidden.is_cuda and teacher_hidden.dtype in (torch.float16, torch.bfloat16):
            projector_autocast = torch.amp.autocast(
                "cuda",
                dtype=teacher_hidden.dtype,
            )
        with projector_autocast:
            projected = self.projector(teacher_hidden)
        if projected.dtype != torch.float32:
            projected = projected.float()
        if self.log_text and teacher_text is not None and self._is_primary_rank():
            for sample_idx, sample_text in enumerate(teacher_text):
                logger.info("Qwen teacher text [%d]: %s", sample_idx, sample_text)
                print(f"Qwen teacher text [{sample_idx}]: {sample_text}", flush=True)
        return {
            "alignment_teacher_embedding": F.normalize(projected, dim=-1),
            "alignment_teacher_text": teacher_text if self.log_text else None,
        }


class TextAlignmentHead(nn.Module):
    def __init__(
        self,
        STUDENT: Dict,
        TEACHER: Optional[Dict] = None,
        compute_teacher: bool = True,
        compute_student_on_eval: bool = True,
    ):
        super().__init__()
        teacher_cfg = copy.deepcopy(TEACHER)
        student_cfg = copy.deepcopy(STUDENT)
        self.teacher = (
            instantiate(teacher_cfg, _recursive_=False) if teacher_cfg is not None else None
        )
        student_embedding_dim = student_cfg.get("embedding_dim", None)
        if student_embedding_dim is None or student_embedding_dim <= 0:
            teacher_output_dim = getattr(self.teacher, "output_embedding_dim", None)
            if teacher_output_dim is None:
                raise ValueError(
                    "STUDENT.embedding_dim must be positive unless the teacher exposes "
                    "output_embedding_dim for automatic alignment"
                )
            student_cfg["embedding_dim"] = teacher_output_dim

        self.student = instantiate(student_cfg, _recursive_=False)
        teacher_output_dim = getattr(self.teacher, "output_embedding_dim", None)
        student_output_dim = getattr(self.student, "embedding_dim", None)
        if (
            teacher_output_dim is not None
            and student_output_dim is not None
            and teacher_output_dim != student_output_dim
        ):
            raise ValueError(
                "Alignment student and teacher output dims must match, got "
                f"{student_output_dim} vs {teacher_output_dim}"
            )
        self.compute_teacher = compute_teacher
        self.compute_student_on_eval = compute_student_on_eval

    @property
    def extra_attention_blocks(self):
        return getattr(self.student, "extra_attention_blocks", None)

    @property
    def extra_attention_use_qk_norm(self):
        return getattr(self.student, "extra_attention_use_qk_norm", False)

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        **kwargs,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if not self.training and not self.compute_student_on_eval:
            student_output = {}
        else:
            student_output = self.student(aggregated_tokens_list[-1], patch_start_idx)

        if self.teacher is None or not self.compute_teacher:
            teacher_output = {
                "alignment_teacher_embedding": None,
                "alignment_teacher_text": None,
            }
        else:
            teacher_output = self.teacher(images)

        return {
            **student_output,
            **teacher_output,
        }
