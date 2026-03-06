"""Whole-MOVA rCM distillation model.

Wraps the complete MOVA denoiser (video_dit + video_dit_2 + audio_dit +
dual_tower_bridge) into a single ``MOVADenoiser`` nn.Module, then keeps a
frozen *teacher* copy and a trainable *student* copy for consistency
distillation.
"""
from __future__ import annotations

import os
import sys
import glob
import gc
import html
import re
import json
import ftfy
from copy import deepcopy
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Union

import attrs
import imaginaire.utils.distributed

# Monkey Patch: Disable DDP wrapping for this specific model because it uses FSDP internally.
# Wrapping FSDP with DDP causes double memory usage and logic conflicts.
_original_parallel_model_wrapper = imaginaire.utils.distributed.parallel_model_wrapper

def _patched_parallel_model_wrapper(config_ddp, model):
    # Use class name check to avoid circular import issues
    if model.__class__.__name__ == "IT2AVDistillModel_rCM":
        log.info("Skipping DDP wrapper for IT2AVDistillModel_rCM (Using Internal FSDP)")
        return model
    return _original_parallel_model_wrapper(config_ddp, model)

imaginaire.utils.distributed.parallel_model_wrapper = _patched_parallel_model_wrapper

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
# from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, CPUOffload, ShardingStrategy

from torch.distributed._tensor.api import DTensor
from transformers import T5TokenizerFast, UMT5EncoderModel

from imaginaire.model import ImaginaireModel
from imaginaire.utils import log
from imaginaire.lazy_config import instantiate as lazy_instantiate
from rcm.utils.denoiser_scaling import RectifiedFlow_TrigFlowWrapper
from rcm.utils.lognormal import LogNormal
from rcm.utils.optim_instantiate_dtensor import get_base_scheduler
from rcm.utils.fsdp_helper import hsdp_device_mesh
from rcm.utils.dtensor_helper import broadcast_dtensor_model_states
from rcm.utils.torch_future import clip_grad_norm_


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text

class _IdentityTokenizer(torch.nn.Module):
    """Minimal tokenizer stub to satisfy TurboDiffusion callback contract."""

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @torch.no_grad()
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _maybe_add_mova_to_syspath() -> None:
    mova_root = os.environ.get("MOVA_ROOT", "")
    if mova_root and os.path.isdir(mova_root) and mova_root not in sys.path:
        sys.path.append(mova_root)
        return
    # Assuming environment is already set up correctly or MOVA_ROOT is provided.



# ═══════════════════════════════════════════════════════════════════════════
# MOVADenoiser — unified nn.Module wrapping the complete MOVA denoiser
# ═══════════════════════════════════════════════════════════════════════════

class MOVADenoiser(torch.nn.Module):
    """Single nn.Module that owns every trainable component of MOVA.

    Sub-modules:
        video_dit        – high-noise video DiT  (e.g. 30 blocks)
        video_dit_2      – low-noise  video DiT  (e.g. 10 blocks)
        audio_dit        – audio DiT             (e.g. 30 blocks)
        bridge           – DualTowerConditionalBridge (cross-attn, fires at
                           specific layers inside the block loop)

    ``forward()`` reproduces ``MOVA.inference_single_step`` +
    ``forward_dual_tower_dit`` exactly:

        for layer_idx in range(min_layers):
            if bridge.should_interact(layer_idx, "a2v"):
                visual_x, audio_x = bridge(layer_idx, visual_x, audio_x, ...)
            visual_x = video_dit.blocks[layer_idx](visual_x, ...)
            audio_x  = audio_dit.blocks[layer_idx](audio_x, ...)
        for layer_idx in range(min_layers, visual_layers):
            visual_x = video_dit.blocks[layer_idx](visual_x, ...)
    """

    def __init__(
        self,
        video_dit: torch.nn.Module,
        video_dit_2: torch.nn.Module,
        audio_dit: torch.nn.Module,
        bridge: torch.nn.Module,
        boundary_ratio: float = 0.9,
        num_train_timesteps: float = 1000.0,
        video_fps: float = 24.0,
    ):
        super().__init__()
        self.video_dit = video_dit
        self.video_dit_2 = video_dit_2
        self.audio_dit = audio_dit
        self.bridge = bridge
        self.boundary_ratio = boundary_ratio
        self.num_train_timesteps = num_train_timesteps
        self.video_fps = video_fps

    # ------------------------------------------------------------------
    # Select video_dit vs video_dit_2 based on noise level
    # ------------------------------------------------------------------
    def _select_video_dit(self, c_noise: Tensor) -> torch.nn.Module:
        """MOVA switches from ``video_dit`` to ``video_dit_2`` when
        ``timestep < boundary_ratio * num_train_timesteps``.
        """
        boundary = self.boundary_ratio * self.num_train_timesteps
        t_val = c_noise.reshape(-1)[0].item()
        return self.video_dit_2 if t_val < boundary else self.video_dit

    # ------------------------------------------------------------------
    # forward — full joint MOVA denoiser pass
    # ------------------------------------------------------------------
    def forward(
        self,
        v_xt: Tensor,          # already scaled by c_in; [B, C_v(+cond), T, H, W]
        a_xt: Tensor,          # already scaled by c_in; [B, C_a, L_a]
        v_timestep: Tensor,    # c_noise for video (used as timestep)
        a_timestep: Tensor,    # c_noise for audio
        text_emb: Tensor,      # [B, seq_len, text_dim]
        ref_latent: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Returns ``(visual_output, audio_output)`` — raw denoiser outputs
        in the original latent shape (before c_skip / c_out)."""
        from mova.diffusion.models import sinusoidal_embedding_1d

        video_dit = self._select_video_dit(v_timestep)
        audio_dit = self.audio_dit
        bridge = self.bridge

        bsz = v_xt.shape[0]

        # ---- 1. Condition concat for video ----
        #   In the dataset, ref_latent = [mask_lat_size, latent_condition]
        #   concatenated on channels.  video_dit.patch_embedding expects
        #   in_channels = latent_ch + condition_ch.
        
        if v_xt.ndim == 5 and ref_latent is not None:
            cond = self._prepare_condition(ref_latent, v_xt, video_dit)
            if cond is not None:
                visual_latents = torch.cat([v_xt, cond], dim=1)
            else:
                visual_latents = v_xt
        else:
            visual_latents = v_xt

        audio_latents = a_xt
        visual_context = audio_context = text_emb

        # Flatten timesteps to [B]
        v_t = v_timestep.reshape(bsz).to(torch.float32)
        a_t = a_timestep.reshape(bsz).to(torch.float32)

        # ---- 2. Time embeddings (float32 → model_dtype) ----
        #   pipeline_mova.py L536-547
        with torch.autocast("cuda", dtype=torch.float32):
            visual_t = video_dit.time_embedding(
                sinusoidal_embedding_1d(video_dit.freq_dim, v_t),
            )
            visual_t_mod = video_dit.time_projection(visual_t).unflatten(
                1, (6, video_dit.dim),
            )
            audio_t = audio_dit.time_embedding(
                sinusoidal_embedding_1d(audio_dit.freq_dim, a_t),
            )
            audio_t_mod = audio_dit.time_projection(audio_t).unflatten(
                1, (6, audio_dit.dim),
            )

        model_dtype = video_dit.dtype  # bf16
        visual_t     = visual_t.to(model_dtype)
        visual_t_mod = visual_t_mod.to(model_dtype)
        audio_t      = audio_t.to(model_dtype)
        audio_t_mod  = audio_t_mod.to(model_dtype)

        # ---- 3. Text embeddings ----
        #   pipeline_mova.py L550-551
        visual_context_emb = video_dit.text_embedding(visual_context)
        audio_context_emb  = audio_dit.text_embedding(audio_context)

        # ---- 4. Patchify ----
        #   pipeline_mova.py L553-584
        visual_latents = visual_latents.to(model_dtype)
        audio_latents  = audio_latents.to(model_dtype)

        visual_x, (t, h, w) = video_dit.patchify(visual_latents)
        grid_size = (t, h, w)

        vf = tuple(freq.to(visual_x.device) for freq in video_dit.freqs)
        visual_freqs = torch.cat([
            vf[0][:t].view(t, 1, 1, -1).expand(t, h, w, -1),
            vf[1][:h].view(1, h, 1, -1).expand(t, h, w, -1),
            vf[2][:w].view(1, 1, w, -1).expand(t, h, w, -1),
        ], dim=-1).reshape(t * h * w, 1, -1).to(visual_x.device)

        audio_x, (f,) = audio_dit.patchify(audio_latents, None)
        audio_freqs = torch.cat([
            audio_dit.freqs[0][:f].view(f, -1),
            audio_dit.freqs[1][:f].view(f, -1),
            audio_dit.freqs[2][:f].view(f, -1),
        ], dim=-1).reshape(f, 1, -1).to(audio_x.device)

        # ---- 5. Cross-modal RoPE ----
        #   forward_dual_tower_dit L641-648
        if bridge.apply_cross_rope:
            visual_rope_cos_sin, audio_rope_cos_sin = \
                bridge.build_aligned_freqs(
                    video_fps=self.video_fps,
                    grid_size=grid_size,
                    audio_steps=audio_x.shape[1],
                    device=visual_x.device,
                    dtype=visual_x.dtype,
                )
        else:
            visual_rope_cos_sin = None
            audio_rope_cos_sin = None

        # ---- 5.5 Sequence Parallel Setup ----
        sp_enabled = False
        sp_group = None
        sp_rank = 0
        sp_size = 1
        visual_pad_len = 0
        audio_pad_len = 0
        
        try:
            from megatron.core import parallel_state
            if parallel_state.is_initialized():
                sp_size = parallel_state.get_context_parallel_world_size()
                if sp_size > 1:
                    sp_rank = parallel_state.get_context_parallel_rank()
                    sp_group = parallel_state.get_context_parallel_group()
                    sp_enabled = True
        except ImportError:
            pass

        if sp_enabled:
            from mova.distributed.functional import _sp_split_tensor, _sp_split_tensor_dim_0
            visual_x, visual_chunk_len, visual_pad_len, _ = _sp_split_tensor(visual_x, sp_size=sp_size, sp_rank=sp_rank)
            audio_x, audio_chunk_len, audio_pad_len, _ = _sp_split_tensor(audio_x, sp_size=sp_size, sp_rank=sp_rank)
            visual_freqs, _, _, _ = _sp_split_tensor_dim_0(visual_freqs, sp_size=sp_size, sp_rank=sp_rank)
            audio_freqs, _, _, _ = _sp_split_tensor_dim_0(audio_freqs, sp_size=sp_size, sp_rank=sp_rank)
            if visual_rope_cos_sin is not None:
                visual_rope_cos_sin = [
                    _sp_split_tensor(rope_cos_sin, sp_size=sp_size, sp_rank=sp_rank)[0]
                    for rope_cos_sin in visual_rope_cos_sin
                ]
            if audio_rope_cos_sin is not None:
                audio_rope_cos_sin = [
                    _sp_split_tensor(rope_cos_sin, sp_size=sp_size, sp_rank=sp_rank)[0]
                    for rope_cos_sin in audio_rope_cos_sin
                ]
            if len(visual_t_mod.shape) == 4:
                visual_t_mod, _, _, _ = _sp_split_tensor(visual_t_mod, sp_size=sp_size, sp_rank=sp_rank)

        # ---- 6. Joint block loop ----
        #   forward_dual_tower_dit L676-702
        #
        #   The bridge fires at specific layers (e.g. 0-9 for "shallow_focus")
        #   doing BIDIRECTIONAL cross-attention (a2v + v2a) in a single call.
        #   Then both towers advance one block.
        min_layers    = min(len(video_dit.blocks), len(audio_dit.blocks))
        visual_layers = len(video_dit.blocks)

        for layer_idx in range(min_layers):
            visual_block = video_dit.blocks[layer_idx]
            audio_block  = audio_dit.blocks[layer_idx]

            # Bridge cross-attention (bidirectional: a2v + v2a)
            if bridge.should_interact(layer_idx, "a2v"):
                visual_x, audio_x = bridge(
                    layer_idx,
                    visual_x,
                    audio_x,
                    x_freqs=visual_rope_cos_sin,
                    y_freqs=audio_rope_cos_sin,
                    a2v_condition_scale=None,
                    v2a_condition_scale=None,
                    condition_scale=1.0,
                    video_grid_size=grid_size,
                )

            # DiT block forward — DiTBlock.forward(x, context, t_mod, freqs)
            if use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        visual_x = torch.utils.checkpoint.checkpoint(
                            visual_block, visual_x, visual_context_emb,
                            visual_t_mod, visual_freqs,
                            use_reentrant=False,
                        )
                        audio_x = torch.utils.checkpoint.checkpoint(
                            audio_block, audio_x, audio_context_emb,
                            audio_t_mod, audio_freqs,
                            use_reentrant=False,
                        )
                else:
                    visual_x = torch.utils.checkpoint.checkpoint(
                        visual_block, visual_x, visual_context_emb,
                        visual_t_mod, visual_freqs,
                        use_reentrant=False,
                    )
                    audio_x = torch.utils.checkpoint.checkpoint(
                        audio_block, audio_x, audio_context_emb,
                        audio_t_mod, audio_freqs,
                        use_reentrant=False,
                    )
            else:
                visual_x = visual_block(
                    visual_x, visual_context_emb, visual_t_mod, visual_freqs,
                )
                audio_x = audio_block(
                    audio_x, audio_context_emb, audio_t_mod, audio_freqs,
                )

        # Remaining visual-only blocks
        for layer_idx in range(min_layers, visual_layers):
            visual_block = video_dit.blocks[layer_idx]
            if use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        visual_x = torch.utils.checkpoint.checkpoint(
                            visual_block, visual_x, visual_context_emb,
                            visual_t_mod, visual_freqs,
                            use_reentrant=False,
                        )
                else:
                    visual_x = torch.utils.checkpoint.checkpoint(
                        visual_block, visual_x, visual_context_emb,
                        visual_t_mod, visual_freqs,
                        use_reentrant=False,
                    )
            else:
                visual_x = visual_block(
                    visual_x, visual_context_emb, visual_t_mod, visual_freqs,
                )

        # ---- 6.5 Sequence Parallel Gather ----
        if sp_enabled:
            from mova.distributed.functional import _sp_all_gather_avg
            visual_x_full = _sp_all_gather_avg(visual_x, sp_group=sp_group, pad_len=visual_pad_len)
            audio_x_full = _sp_all_gather_avg(audio_x, sp_group=sp_group, pad_len=audio_pad_len)
        else:
            visual_x_full = visual_x
            audio_x_full = audio_x

        # ---- 7. Head & unpatchify ----
        #   pipeline_mova.py L603-607
        visual_output = video_dit.head(visual_x_full, visual_t)
        visual_output = video_dit.unpatchify(visual_output, grid_size)

        audio_output = audio_dit.head(audio_x_full, audio_t)
        audio_output = audio_dit.unpatchify(audio_output, (f,))

        return visual_output.float(), audio_output.float()

    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_condition(
        ref_latent: Tensor, v_xt: Tensor, net: torch.nn.Module,
    ) -> Optional[Tensor]:
        """Build condition tensor (mask + latent_condition) for video DiT."""
        patch = getattr(net, "patch_embedding", None)
        if patch is None: return None
        
        # Robustly get in_channels
        in_ch = getattr(patch, "in_channels", None)
        if in_ch is None and hasattr(patch, "_fsdp_wrapped_module"):
             in_ch = getattr(patch._fsdp_wrapped_module, "in_channels", None)
        
        if in_ch is None: return None
        in_ch = int(in_ch)

        v_ch = int(v_xt.shape[1])
        expected = in_ch - v_ch
        
        if expected <= 0: return None
            
        ref = ref_latent.to(device=v_xt.device, dtype=v_xt.dtype)
        current_ref_ch = ref.shape[1]
        
        # Case 1: Perfect match (e.g. 20 == 20)
        if current_ref_ch == expected:
            return ref
            
        # Case 2: Missing Mask (e.g. Expected 20, Got 16. Missing 4 channels for mask)
        if current_ref_ch == 16 and expected == 20:
            B, _, T, H, W = v_xt.shape
            mask = torch.ones((B, 4, T, H, W), device=v_xt.device, dtype=v_xt.dtype)
            
            if ref.shape[2] == 1 and T > 1:
                 ref = ref.expand(-1, -1, T, -1, -1)
            
            return torch.cat([mask, ref], dim=1)

        return None

    # ------------------------------------------------------------------
    @staticmethod
    def from_mova_pipeline(pipe, video_fps: float = 24.0) -> "MOVADenoiser":
        """Extract the four denoiser sub-networks from a loaded MOVA pipeline
        and wrap them into a single ``MOVADenoiser``."""
        return MOVADenoiser(
            video_dit=pipe.video_dit,
            video_dit_2=pipe.video_dit_2,
            audio_dit=pipe.audio_dit,
            bridge=pipe.dual_tower_bridge,
            boundary_ratio=float(pipe.boundary_ratio),
            num_train_timesteps=float(pipe.scheduler.config.num_train_timesteps),
            video_fps=video_fps,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Config & Batch
# ═══════════════════════════════════════════════════════════════════════════

@attrs.define(slots=False)
class IT2AVDistillConfig_rCM:
    # Hydra composition stubs
    conditioner: Any = None
    tokenizer: Any = None
    ema: Any = None

    # CFG
    teacher_guidance: float = 1.0
    negative_prompt: str = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
        "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
        "残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
        "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    )

    # Data keys
    v_latent_key: str = "v_latents"
    a_latent_key: str = "a_latents"
    ref_latent_key: str = "ref_latents"
    text_embed_key: str = "t5_text_embeddings"

    # Checkpoints
    teacher_ckpt_path: str = ""
    student_ckpt_path: str = ""

    # Precision
    precision: str = "bfloat16"
    rectified_flow_t_scaling_factor: float = 1000.0

    # Activation checkpointing
    use_gradient_checkpointing: bool = True
    use_gradient_checkpointing_offload: bool = False

    # SCM hyper-parameters
    loss_scale: float = 100.0
    tangent_warmup: int = 1000
    sigma_data: float = 1.0
    p_mean: float = -0.8
    p_std: float = 1.6
    fd_type: int = 2
    fd_size: float = 1e-4

    # FSDP
    fsdp_shard_size: int = 40

    # Video FPS for cross-modal RoPE
    video_fps: float = 24.0


@dataclass
class IT2AVBatch:
    v_x0: Tensor
    a_x0: Tensor
    text_emb: Tensor
    ref_latent: Optional[Tensor]


# ═══════════════════════════════════════════════════════════════════════════
# Distillation Model
# ═══════════════════════════════════════════════════════════════════════════

class IT2AVDistillModel_rCM(ImaginaireModel):
    """Whole-MOVA rCM distillation model.

    Holds exactly two ``MOVADenoiser`` instances:

        self.teacher  — frozen, produces the target velocity field
        self.student  — trainable, learns to match teacher in fewer steps

    Every call to ``_denoise_joint`` goes through ``MOVADenoiser.forward()``,
    which runs the full joint dual-tower loop with bridge cross-attention.
    """

    def __init__(self, config: IT2AVDistillConfig_rCM):
        super().__init__()
        self.config = config
        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}

        self.p_G = LogNormal(p_mean=config.p_mean, p_std=config.p_std)
        self.scaling = RectifiedFlow_TrigFlowWrapper(
            config.sigma_data, config.rectified_flow_t_scaling_factor,
        )
        self.tokenizer = _IdentityTokenizer()

        # FSDP mesh
        if config.fsdp_shard_size > 1:
            log.info(f"FSDP shard size: {config.fsdp_shard_size}")
            self.fsdp_device_mesh = hsdp_device_mesh(
                sharding_group_size=config.fsdp_shard_size,
            )
        else:
            self.fsdp_device_mesh = None

        # Import MOVA
        _maybe_add_mova_to_syspath()
        try:
            from mova.diffusion.models import WanModel, WanAudioModel
            from mova.diffusion.models.interactionv2 import DualTowerConditionalBridge
        except ImportError as e:
            raise ImportError(
                "Failed to import MOVA models. Set MOVA_ROOT. "
                f"Original: {e!r}"
            ) from e

        if not config.teacher_ckpt_path:
            raise ValueError("config.teacher_ckpt_path must be set.")

        video_fps = float(getattr(config, "video_fps", 24.0))

        # ---- 1. Load Scheduler Config for num_train_timesteps ----
        num_train_timesteps = 1000.0
        try:
            sched_path = os.path.join(config.teacher_ckpt_path, "scheduler", "scheduler_config.json")
            if os.path.exists(sched_path):
                with open(sched_path, 'r') as f:
                    sched_config = json.load(f)
                num_train_timesteps = float(sched_config.get("num_train_timesteps", 1000.0))
                log.info(f"Loaded num_train_timesteps={num_train_timesteps} from {sched_path}")
        except Exception as e:
            log.warning(f"Could not load scheduler config: {e}, using default 1000.0")

        # ---- 2. Pre-compute negative prompt embedding (Load T5 if needed) ----
        self.neg_text_emb = None
        cfg_scale = float(getattr(config, "teacher_guidance", 1.0))
        if cfg_scale > 1.0:
            neg_prompt = str(getattr(config, "negative_prompt", "")).strip()
            if not neg_prompt:
                raise ValueError("CFG enabled but negative_prompt is empty.")
            
            log.info("Loading T5 for CFG negative prompt...")
            try:
                tokenizer = T5TokenizerFast.from_pretrained(
                    config.teacher_ckpt_path, subfolder="tokenizer"
                )
                text_encoder = UMT5EncoderModel.from_pretrained(
                    config.teacher_ckpt_path, subfolder="text_encoder", 
                        torch_dtype=self.precision
                    )
                
                # Compute embedding using local helper
                with torch.no_grad():
                    self.neg_text_emb = self._compute_t5_embeds(
                        text_encoder, tokenizer, neg_prompt, device="cpu"
                    ).to(dtype=self.precision, device="cpu")
                
                del tokenizer
                del text_encoder
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                log.info("Negative prompt embedding computed, T5 unloaded.")
            except Exception as e:
                log.warning(f"Failed to load T5 for negative prompt: {e}")
                # Fallback or fail? Fail is safer if CFG is requested.
                raise RuntimeError(f"CFG enabled but failed to load T5: {e}") from e

        # ---- 3. Load Model Components Sequentially ----
        log.info(f"Loading MOVA components from {config.teacher_ckpt_path}")
        
        log.info("Loading Video DiT...")
        video_dit = WanModel.from_pretrained(
            config.teacher_ckpt_path, subfolder="video_dit", torch_dtype=self.precision
        )
        
        log.info("Loading Video DiT 2...")
        video_dit_2 = WanModel.from_pretrained(
            config.teacher_ckpt_path, subfolder="video_dit_2", torch_dtype=self.precision
        )
        
        log.info("Loading Audio DiT...")
        audio_dit = WanAudioModel.from_pretrained(
            config.teacher_ckpt_path, subfolder="audio_dit", torch_dtype=self.precision
        )
        
        log.info("Loading Bridge...")
        bridge = DualTowerConditionalBridge.from_pretrained(
            config.teacher_ckpt_path, subfolder="dual_tower_bridge", torch_dtype=self.precision
        )

        # Teacher (frozen)
        self.teacher = MOVADenoiser(
            video_dit=video_dit,
            video_dit_2=video_dit_2,
            audio_dit=audio_dit,
            bridge=bridge,
            boundary_ratio=0.9,
            num_train_timesteps=num_train_timesteps,
            video_fps=video_fps,
        )
        self.teacher.eval().requires_grad_(False)

        # Student (trainable) — deep-copy of teacher
        # NOTE: Deepcopying huge models doubles memory usage.
        # If possible, we should reload from disk or use cpu offload.
        # For now, we assume deepcopy is fine if we fit in RAM, but it might be tight.
        # Given we unloaded T5/VAE, we have ~10GB+ headroom.
        self.student = deepcopy(self.teacher).train().requires_grad_(True)

        # ---- Optionally overwrite student weights ----
        if config.student_ckpt_path and config.student_ckpt_path != config.teacher_ckpt_path:
            log.info(f"Overwriting student weights from {config.student_ckpt_path}")
            
            
            # --- New logic: monolithic loading with prefixes from turbo_t2av_core ---
            
            # Force path to point to turbo_t2av_core
            target_path = config.student_ckpt_path
            if not target_path.endswith("turbo_t2av_core"):
                 target_path = os.path.join(target_path, "turbo_t2av_core")
            
            if not os.path.isdir(target_path):
                 raise ValueError(f"Required folder 'turbo_t2av_core' not found in {config.student_ckpt_path}")
            
            weight_files = sorted(glob.glob(os.path.join(target_path, "*.safetensors")))
            if not weight_files:
                weight_files = sorted(glob.glob(os.path.join(target_path, "*.pt")))

            if not weight_files:
                raise ValueError(f"No .safetensors or .pt weights found in {target_path}!")

            log.info(f"Found {len(weight_files)} weight files in {target_path}. Loading...")
            full_state_dict = {}
            
            try:
                from safetensors.torch import load_file as load_safetensors
            except ImportError:
                load_safetensors = None

            for wf in weight_files:
                if wf.endswith(".safetensors"):
                    if load_safetensors:
                        part = load_safetensors(wf)
                    else:
                        log.warning("safetensors library missing. Trying torch.load on .safetensors file.")
                        part = torch.load(wf, map_location="cpu")
                else:
                    part = torch.load(wf, map_location="cpu")
                full_state_dict.update(part)

            # Define prefix mapping
            sections = {
                "video_high": self.student.video_dit,
                "video_low":  self.student.video_dit_2,
                "audio":      self.student.audio_dit,
                "bridge":     self.student.bridge,
            }

            log.info("🔄 开始根据前缀拆分并加载权重...")
            for prefix, model_instance in sections.items():
                log.info(f"📦 正在处理 [{prefix}] 模块...")
                sub_state_dict = {}
                prefix_dot = f"{prefix}."
                
                for key, tensor in full_state_dict.items():
                    if key.startswith(prefix_dot):
                        sub_state_dict[key[len(prefix_dot):]] = tensor
                
                if len(sub_state_dict) == 0:
                    raise ValueError(f"⚠️ 错误: 在权重字典中没有找到前缀为 '{prefix}.' 的参数！")
                
                # Auto-reshape: if checkpoint shape != model shape but numel matches, reshape
                model_state = model_instance.state_dict()
                for key in list(sub_state_dict.keys()):
                    ckpt_tensor = sub_state_dict[key]
                    if key in model_state:
                        model_tensor = model_state[key]
                        if ckpt_tensor.shape != model_tensor.shape:
                            if ckpt_tensor.numel() == model_tensor.numel():
                                sub_state_dict[key] = ckpt_tensor.view(model_tensor.shape)
                                log.info(f"  Reshaped {key}: {ckpt_tensor.shape} -> {model_tensor.shape}")
                            else:
                                raise RuntimeError(
                                    f"Shape mismatch for {key}: checkpoint {ckpt_tensor.shape} (numel={ckpt_tensor.numel()}) "
                                    f"vs model {model_tensor.shape} (numel={model_tensor.numel()})"
                                )
                    
                model_instance.load_state_dict(sub_state_dict, strict=True)
                log.info(f"✅ [{prefix}] 成功加载了 {len(sub_state_dict)} 个张量！")

            log.info("\n🎉 All prefix modules loaded from turbo_t2av_core.")
            del full_state_dict
            gc.collect()


        # ---- Replace attention for Sequence Parallelism ----
        try:
            from megatron.core import parallel_state
            if parallel_state.is_initialized() and parallel_state.get_context_parallel_world_size() > 1:
                from yunchang.kernels import AttnType
                log.info("Replacing attention with USPAttention for Context Parallelism...")
                
                from functools import partial
                from mova.diffusion.models.wan_video_dit import USPAttention
                partial_replace = partial(USPAttention, attn_type=AttnType.FA)
                
                def _replace_attn(denoiser):
                    replaced_cnt = 0
                    for block in denoiser.video_dit.blocks:
                        block.self_attn.attn = partial_replace(block.self_attn.attn.num_heads)
                        replaced_cnt += 1
                    if getattr(denoiser, "video_dit_2", None) is not None:
                        for block in denoiser.video_dit_2.blocks:
                            block.self_attn.attn = partial_replace(block.self_attn.attn.num_heads)
                            replaced_cnt += 1
                    if getattr(denoiser, "audio_dit", None) is not None:
                        for block in denoiser.audio_dit.blocks:
                            block.self_attn.attn = partial_replace(block.self_attn.attn.num_heads)
                            replaced_cnt += 1
                    if getattr(denoiser, "bridge", None) is not None:
                        for conditioner in denoiser.bridge.audio_to_video_conditioners.values():
                            conditioner.inner.attn = partial_replace(conditioner.inner.attn.num_heads)
                            replaced_cnt += 1
                        for conditioner in denoiser.bridge.video_to_audio_conditioners.values():
                            conditioner.inner.attn = partial_replace(conditioner.inner.attn.num_heads)
                            replaced_cnt += 1
                    return replaced_cnt

                c1 = _replace_attn(self.teacher)
                c2 = _replace_attn(self.student)
                log.info(f"Replaced {c1} attention blocks in Teacher and {c2} in Student.")
        except Exception as e:
            log.warning(f"Could not replace attention for SP: {e}")

        # ---- FSDP wrapping (Classic FSDP) ----
        if self.fsdp_device_mesh:
            # Classic FSDP configuration
            mp_policy = MixedPrecision(
                param_dtype=self.precision,  # Use model precision (bf16) instead of forcing fp32
                reduce_dtype=torch.float32, 
                buffer_dtype=self.precision
            )
            
            # Helper to wrap module and replace in parent
            def _wrap(module, cpu_offload=False):
                # Ensure module is on CPU before wrapping if offloading is requested
                if cpu_offload:
                    module = module.to("cpu")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Force use_orig_params=False for stability with complex control flow (distillation)
                # This avoids "BACKWARD_POST" stuck states on small layers.
                try:
                    fsdp_mod = FSDP(
                        module,
                        mixed_precision=mp_policy,
                        cpu_offload=CPUOffload(offload_params=cpu_offload),
                        sharding_strategy=ShardingStrategy.FULL_SHARD if config.fsdp_shard_size > 1 else ShardingStrategy.NO_SHARD,
                        device_id=torch.cuda.current_device(),
                        use_orig_params=False, 
                    )
                    return fsdp_mod
                except TypeError as e:
                    log.warning(f"FSDP init failed: {e}")
                    raise

            def _wrap_mova_components(denoiser, cpu_offload=False):
                # 0. Ensure entire models are on CPU first
                if cpu_offload:
                    for net in [denoiser.video_dit, denoiser.video_dit_2, denoiser.audio_dit, denoiser.bridge]:
                        net.to("cpu")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # 1. Wrap Video/Audio DiT Blocks ONLY
                # We DO NOT wrap sub-components like embeddings individually to avoid deadlock.
                # We DO NOT wrap the video_dit container to avoid 1-D tensor errors.
                # Instead, we wrap the whole denoiser at the end to handle all non-block params.
                for net in [denoiser.video_dit, denoiser.video_dit_2, denoiser.audio_dit]:
                    for i, block in enumerate(net.blocks):
                        net.blocks[i] = _wrap(block, cpu_offload)

                # 2. Wrap Bridge Conditioners and Bridge itself
                # ModuleDict values replacement
                for name, conditioner in denoiser.bridge.audio_to_video_conditioners.items():
                    denoiser.bridge.audio_to_video_conditioners[name] = _wrap(conditioner, cpu_offload)
                for name, conditioner in denoiser.bridge.video_to_audio_conditioners.items():
                    denoiser.bridge.video_to_audio_conditioners[name] = _wrap(conditioner, cpu_offload)
                
                denoiser.bridge = _wrap(denoiser.bridge, cpu_offload)

                # 3. Wrap the ENTIRE Denoiser
                # This FSDP unit will manage all "leftover" params (embeddings, projections, final layers)
                # from video_dit and audio_dit. This ensures they are unsharded once at the start of forward.
                return _wrap(denoiser, cpu_offload)

            # Apply wrapping and update references
            # Teacher: CPU offload when use_gradient_checkpointing_offload=True to save GPU memory (teacher is frozen, no backward)
            teacher_offload = bool(getattr(config, "use_gradient_checkpointing_offload", False))
            self.teacher = _wrap_mova_components(self.teacher, cpu_offload=teacher_offload)
            
            # Student: CPU offload when use_gradient_checkpointing_offload=True (slower but saves GPU memory)
            student_offload = bool(getattr(config, "use_gradient_checkpointing_offload", False))
            self.student = _wrap_mova_components(self.student, cpu_offload=student_offload)
            
            # We don't need explicit broadcast_dtensor_model_states for classic FSDP 
            # as it handles sync internally during init/forward.

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_t5_embeds(
        text_encoder,
        tokenizer,
        prompt: Union[str, List[str]] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Helper to compute T5 embeddings without loading full pipeline."""
        device = device or text_encoder.device
        dtype = text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )
        return prompt_embeds

    # ------------------------------------------------------------------
    @staticmethod
    def _load_submodule(
        ckpt_root: str, subfolder: str, target: torch.nn.Module,
    ) -> None:
        path = os.path.join(ckpt_root, subfolder)
        if not os.path.isdir(path):
            log.warning(f"No {subfolder} in {ckpt_root}, keeping teacher wts.")
            return
        try:
            loaded = type(target).from_pretrained(
                ckpt_root, subfolder=subfolder,
            )
            target.load_state_dict(loaded.state_dict())
            del loaded
            log.info(f"Loaded {subfolder} from student path.")
        except Exception as e:
            log.warning(f"Failed to load {subfolder}: {e}")

    # ------------------------------------------------------------------
    # Batch parsing
    # ------------------------------------------------------------------
    def _parse_batch(self, data_batch: Dict[str, Any]) -> IT2AVBatch:
        def _sq(x: Tensor) -> Tensor:
            if isinstance(x, torch.Tensor) and x.ndim >= 2 and x.shape[1] == 1:
                return x[:, 0]
            return x

        v = _sq(data_batch[self.config.v_latent_key]).to(**self.tensor_kwargs)
        a = _sq(data_batch[self.config.a_latent_key]).to(**self.tensor_kwargs)
        t = _sq(data_batch[self.config.text_embed_key]).to(**self.tensor_kwargs)
        r = data_batch.get(self.config.ref_latent_key)
        if r is not None:
            r = _sq(r).to(**self.tensor_kwargs)
        return IT2AVBatch(v_x0=v, a_x0=a, text_emb=t, ref_latent=r)

    # ------------------------------------------------------------------
    # Time utilities
    # ------------------------------------------------------------------
    def _broadcast_time(self, time_B: Tensor, ndim: int) -> Tensor:
        if ndim == 5:
            return time_B.view(-1, 1, 1, 1, 1)
        if ndim == 3:
            return time_B.view(-1, 1, 1)
        raise ValueError(f"Unsupported ndim={ndim}")

    def _draw_time(self, batch_size: int, multiplier: float) -> Tensor:
        sigma = self.p_G(batch_size) * multiplier
        return torch.arctan(sigma).double()

    # ------------------------------------------------------------------
    # Joint denoise: TrigFlow scaling → MOVADenoiser.forward → x0 / F
    # ------------------------------------------------------------------
    def _denoise_joint(
        self,
        v_xt: Tensor,
        a_xt: Tensor,
        v_time_B: Tensor,
        a_time_B: Tensor,
        text_emb: Tensor,
        denoiser: MOVADenoiser,
        ref_latent: Optional[Tensor] = None,
    ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """Returns ``((v_x0, v_F), (a_x0, a_F))``."""

        # TrigFlow scaling
        v_tv = self._broadcast_time(v_time_B, v_xt.ndim)
        v_skip, v_out, v_in, v_noise = self.scaling(trigflow_t=v_tv)

        a_tv = self._broadcast_time(a_time_B, a_xt.ndim)
        a_skip, a_out, a_in, a_noise = self.scaling(trigflow_t=a_tv)

        # Gradient checkpointing only for student
        use_ckpt = (
            self.training
            and denoiser is self.student
            and bool(getattr(self.config, "use_gradient_checkpointing", True))
        )

        amp_ctx = (
            torch.autocast("cuda", dtype=self.precision)
            if self.precision in (torch.float16, torch.bfloat16)
            else nullcontext()
        )

        with amp_ctx:
            v_scaled = (v_xt * v_in).to(**self.tensor_kwargs)
            a_scaled = (a_xt * a_in).to(**self.tensor_kwargs)

            # ★ Single call to the unified MOVADenoiser ★
            use_offload = bool(getattr(self.config, "use_gradient_checkpointing_offload", False))
            v_raw, a_raw = denoiser(
                v_scaled, a_scaled,
                v_noise, a_noise,
                text_emb,
                ref_latent=ref_latent,
                use_gradient_checkpointing=use_ckpt,
                use_gradient_checkpointing_offload=use_offload,
            )

        # Reconstruct x0 and velocity F
        v_x0 = (v_skip * v_xt + v_out * v_raw).to(dtype=v_xt.dtype)
        a_x0 = (a_skip * a_xt + a_out * a_raw).to(dtype=a_xt.dtype)

        vc, vs = torch.cos(v_tv), torch.sin(v_tv)
        v_F = (vc * v_xt - v_x0) / vs

        ac, a_s = torch.cos(a_tv), torch.sin(a_tv)
        a_F = (ac * a_xt - a_x0) / a_s

        return (v_x0, v_F), (a_x0, a_F)

    # ------------------------------------------------------------------
    # training_step
    # ------------------------------------------------------------------
    def training_step(
        self, data_batch: Dict[str, Tensor], iteration: int = 0,
    ):
        batch = self._parse_batch(data_batch)

        bsz = batch.v_x0.shape[0]

        # Sample trigflow times
        v_mult = (float(np.sqrt(batch.v_x0.shape[2]))
                  if batch.v_x0.ndim == 5 else 1.0)
        v_time = self._draw_time(bsz, v_mult)
        a_time = self._draw_time(bsz, 1.0)

        # Build noisy latents
        v_tv = self._broadcast_time(v_time, batch.v_x0.ndim)
        a_tv = self._broadcast_time(a_time, batch.a_x0.ndim)
        vc, vs = torch.cos(v_tv), torch.sin(v_tv)
        ac, a_s = torch.cos(a_tv), torch.sin(a_tv)
        v_xt = (batch.v_x0.float() * vc + torch.randn_like(batch.v_x0.float()) * vs).to(batch.v_x0.dtype)
        a_xt = (batch.a_x0.float() * ac + torch.randn_like(batch.a_x0.float()) * a_s).to(batch.a_x0.dtype)

        # ---- Teacher (no grad) ----
        with torch.no_grad():
            cfg = float(getattr(self.config, "teacher_guidance", 1.0))
            if cfg > 1.0:
                if getattr(self, "neg_text_emb", None) is None:
                    raise RuntimeError("CFG enabled but neg_text_emb missing.")
                neg = self.neg_text_emb
                if neg.ndim == 2:
                    neg = neg.unsqueeze(0)
                neg_B = neg.expand(bsz, -1, -1).to(**self.tensor_kwargs)

                (_, vFc), (_, aFc) = self._denoise_joint(
                    v_xt, a_xt, v_time, a_time, batch.text_emb,
                    self.teacher, ref_latent=batch.ref_latent,
                )
                (_, vFu), (_, aFu) = self._denoise_joint(
                    v_xt, a_xt, v_time, a_time, neg_B,
                    self.teacher, ref_latent=batch.ref_latent,
                )
                v_F_teacher = vFu + cfg * (vFc - vFu)
                a_F_teacher = aFu + cfg * (aFc - aFu)
            else:
                (_, v_F_teacher), (_, a_F_teacher) = self._denoise_joint(
                    v_xt, a_xt, v_time, a_time, batch.text_emb,
                    self.teacher, ref_latent=batch.ref_latent,
                )

        # ---- Student ----
        (v_x0_s, v_Ft), (a_x0_s, a_Ft) = self._denoise_joint(
            v_xt, a_xt, v_time, a_time, batch.text_emb,
            self.student, ref_latent=batch.ref_latent,
        )
        v_Fsg = v_Ft.detach()
        a_Fsg = a_Ft.detach()

        # ---- SCM loss (fd_type=2) ----
        w = min(1.0, float(iteration) / float(self.config.tangent_warmup))
        if self.config.fd_type != 2:
            raise NotImplementedError("Only fd_type=2.")
        h = float(self.config.fd_size)

        v_xt2 = np.cos(h) * v_xt - np.sin(h) * v_F_teacher
        a_xt2 = np.cos(h) * a_xt - np.sin(h) * a_F_teacher

        with torch.no_grad():
            (_, vF2), (_, aF2) = self._denoise_joint(
                v_xt2, a_xt2, v_time - h, a_time - h, batch.text_emb,
                self.student, ref_latent=batch.ref_latent,
            )

        # Video
        dv = (np.cos(h) * v_Ft.detach() - vF2) / np.sin(h)
        tv = vc * vs * dv
        gv = -vc * torch.sqrt(1 - w**2 * vs**2) * (v_Fsg - v_F_teacher) - w * (vc * vs * v_xt + tv)
        loss_v = ((v_Ft - v_Fsg - gv)**2).sum(dim=tuple(range(1, v_Ft.ndim))).mean()

        # Audio
        da = (np.cos(h) * a_Ft.detach() - aF2) / np.sin(h)
        ta = ac * a_s * da
        ga = -ac * torch.sqrt(1 - w**2 * a_s**2) * (a_Fsg - a_F_teacher) - w * (ac * a_s * a_xt + ta)
        loss_a = ((a_Ft - a_Fsg - ga)**2).sum(dim=tuple(range(1, a_Ft.ndim))).mean()

        loss = self.config.loss_scale * (loss_v + loss_a)

        return {
            "loss_v": loss_v.detach(), "loss_a": loss_a.detach(),
            "v_time": v_time.detach(), "a_time": a_time.detach(),
            "v_xt": v_xt.detach(),     "a_xt": a_xt.detach(),
            "v_x0_pred": v_x0_s.detach(), "a_x0_pred": a_x0_s.detach(),
        }, loss

    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Utilities for TurboDiffusion trainer / callbacks
    # ------------------------------------------------------------------
    def model_param_stats(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        learn = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_param_num": total, "total_learnable_param_num": learn}

    def is_image_batch(self, data_batch: Dict[str, Tensor]) -> bool:
        return False

    def init_optimizer_scheduler(self, optimizer_config, scheduler_config):
        optimizer = lazy_instantiate(optimizer_config, model=self)
        scheduler = get_base_scheduler(optimizer, self, scheduler_config)
        return optimizer, scheduler

    def cuda(self, device=None):
        if self.fsdp_device_mesh:
            # In FSDP mode with CPU Offload, teacher MUST remain on CPU.
            # Student is also handled by FSDP, so we shouldn't manually move it either.
            # We only move the non-FSDP parts (if any).
            # But IT2AVDistillModel_rCM itself is usually not wrapped, only its sub-modules.
            # So we only move buffers/params that are NOT in teacher/student.
            
            # Move self (the container) to cuda, but skip teacher/student if they are FSDP wrapped
            # This is tricky because .cuda() is recursive.
            
            # Strategy: Move only what's necessary.
            if self.neg_text_emb is not None:
                self.neg_text_emb = self.neg_text_emb.cuda(device)
            # Scaling wrapper might have buffers?
            # self.scaling... usually stateless or pure buffers
            
            # We DO NOT call super().cuda() or self.teacher.cuda()
            return self
        
        return super().cuda(device)

    def to(self, *args, **kwargs):
        if self.fsdp_device_mesh:
            # Check if target device is cuda
            device = None
            for arg in args:
                if isinstance(arg, (torch.device, str)):
                    device = torch.device(arg)
                    break
            if kwargs.get('device') is not None:
                device = torch.device(kwargs.get('device'))
            
            if device is not None and device.type == 'cuda':
                # Similar to .cuda(), prevent moving teacher/student
                if self.neg_text_emb is not None:
                    self.neg_text_emb = self.neg_text_emb.to(*args, **kwargs)
                return self
        
        return super().to(*args, **kwargs)

    def on_train_start(
        self, memory_format: torch.memory_format = torch.preserve_format,
    ) -> None:
        # Teacher is FSDP-wrapped with CPU Offload, so we MUST NOT move it to CUDA manually.
        # It will handle moving params to GPU on-the-fly during forward pass.
        # Student is also FSDP wrapped, so we generally trust FSDP to handle device placement.
        
        if float(getattr(self.config, "teacher_guidance", 1.0)) > 1.0:
            if getattr(self, "neg_text_emb", None) is None:
                raise RuntimeError("CFG enabled but neg_text_emb missing.")
            self.neg_text_emb = self.neg_text_emb.to(**self.tensor_kwargs)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def clip_grad_norm_(
        self, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False, foreach: Optional[bool] = None,
    ):
        params = [p for p in self.student.parameters() if p.grad is not None]
        if not params:
            return torch.tensor(0.0)
        for p in params:
            torch.nan_to_num(p.grad, nan=0, posinf=0, neginf=0, out=p.grad)
        return clip_grad_norm_(
            params, max_norm=max_norm, norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite, foreach=foreach,
        ).cpu()
