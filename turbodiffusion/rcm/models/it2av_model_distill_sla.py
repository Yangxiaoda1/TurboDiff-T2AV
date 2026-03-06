# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import sys
from copy import deepcopy
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import attrs
import torch
import torch.distributed as dist
from torch import Tensor

from imaginaire.model import ImaginaireModel
from imaginaire.utils import log
from imaginaire.lazy_config import instantiate as lazy_instantiate
from rcm.utils.optim_instantiate_dtensor import get_base_scheduler


class _IdentityTokenizer(torch.nn.Module):
    """A minimal tokenizer for callbacks (see it2av_model_distill_rcm)."""

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @torch.no_grad()
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _maybe_add_mova_to_syspath() -> None:
    """Make `import mova` available in TurboDiffusion training.

    Prefer setting environment variable MOVA_ROOT to the MOVA project directory.
    Fallback: try to locate MOVA as a sibling of TurboDiffusion in monorepo layout.
    """
    mova_root = os.environ.get("MOVA_ROOT", "")
    if mova_root and os.path.isdir(mova_root):
        if mova_root not in sys.path:
            sys.path.append(mova_root)
        return

    # Fallback: <...>/TurboDiffusion/turbodiffusion/rcm/models -> <...>/TurboDiffusion -> <...>/MOVA
    here = os.path.dirname(os.path.abspath(__file__))
    turbo_root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
    candidate = os.path.join(os.path.dirname(turbo_root), "MOVA")
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.append(candidate)


def _truncate_blocks(module: torch.nn.Module, num_layers: int) -> None:
    """Truncate `module.blocks` in-place if present.

    This is a simple speed-oriented student construction: keep the first N blocks.
    """
    if num_layers <= 0:
        return
    if not hasattr(module, "blocks"):
        return
    blocks = getattr(module, "blocks")
    if not isinstance(blocks, torch.nn.ModuleList):
        return
    if num_layers >= len(blocks):
        return
    setattr(module, "blocks", torch.nn.ModuleList(list(blocks)[:num_layers]))


@attrs.define(slots=False)
class IT2AVDistillConfig_SLA:
    # NOTE: Placeholder fields for Hydra default groups.
    conditioner: Any = None
    tokenizer: Any = None
    ema: Any = None

    # --- dataset keys (coming from WebDataset rename_map) ---
    v_latent_key: str = "v_latents"
    a_latent_key: str = "a_latents"
    ref_latent_key: str = "ref_latents"
    text_embed_key: str = "t5_text_embeddings"
    prompt_key: str = "prompts"

    # --- teacher checkpoint (MOVA diffusers-style directory) ---
    mova_ckpt_path: str = ""

    # --- training / numerical ---
    precision: str = "bfloat16"
    rectified_flow_t_scaling_factor: float = 1000.0

    # --- distillation options ---
    teacher_guidance: float = 0.0  # reserved; keep 0.0 unless you add negative embeddings
    student_num_layers_video: int = 0  # 0 -> keep all teacher layers
    student_num_layers_audio: int = 0  # 0 -> keep all teacher layers

    # --- conditioning options ---
    use_ref_latent: bool = False  # reserved; current student ignores ref_latents unless you modify model forward


@dataclass
class IT2AVBatch:
    v_x0: Tensor
    a_x0: Tensor
    text_emb: Tensor
    ref_latent: Optional[Tensor]


class IT2AVDistillModel_SLA(ImaginaireModel):
    """SLA-style online distillation for MOVA (IT2AV).

    This model loads a MOVA checkpoint as the teacher, then trains a student video DiT and audio DiT
    to match the teacher denoiser outputs on random intermediate states x_t.

    Notes:
    - This is intentionally minimal: it distills video/audio towers independently (no cross-modal bridge).
    - It assumes the dataset provides `embed.pt` as the text embedding already (shape [B, L, D]).
    - `ref_latents` is currently not wired into the DiT forward; keep it for future extensions.
    """

    def __init__(self, config: IT2AVDistillConfig_SLA):
        super().__init__()
        self.config = config
        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}
        self.tokenizer = _IdentityTokenizer()

        _maybe_add_mova_to_syspath()
        try:
            from mova.diffusion.pipelines.pipeline_mova import MOVA  # type: ignore
        except Exception as e:
            raise ImportError(
                "Failed to import MOVA. Set environment variable MOVA_ROOT to your MOVA project directory "
                "(the one containing `mova/`). Original error: " + repr(e)
            ) from e

        if not self.config.mova_ckpt_path:
            raise ValueError("config.mova_ckpt_path must be set to the MOVA checkpoint directory (e.g. MOVA-360p).")

        log.info(f"Loading MOVA teacher from {self.config.mova_ckpt_path}")
        # Load full pipeline to ensure the DiT configs/weights match checkpoint.
        # We only keep the two DiTs for distillation.
        pipe = MOVA.from_pretrained(self.config.mova_ckpt_path, torch_dtype=self.precision)

        self.teacher_video = pipe.video_dit.eval().requires_grad_(False)
        self.teacher_audio = pipe.audio_dit.eval().requires_grad_(False)

        # Student initialization: start from teacher weights, then optionally truncate layers.
        self.student_video = deepcopy(self.teacher_video).train().requires_grad_(True)
        self.student_audio = deepcopy(self.teacher_audio).train().requires_grad_(True)
        _truncate_blocks(self.student_video, self.config.student_num_layers_video)
        _truncate_blocks(self.student_audio, self.config.student_num_layers_audio)

        # Free unused large modules from pipeline to save VRAM/CPU RAM.
        del pipe

    def _parse_batch(self, data_batch: Dict[str, Any]) -> IT2AVBatch:
        def _squeeze_b1(x: Tensor) -> Tensor:
            if isinstance(x, torch.Tensor) and x.ndim >= 2 and x.shape[1] == 1:
                return x[:, 0]
            return x

        v_x0 = _squeeze_b1(data_batch[self.config.v_latent_key]).to(**self.tensor_kwargs)
        a_x0 = _squeeze_b1(data_batch[self.config.a_latent_key]).to(**self.tensor_kwargs)
        text_emb = _squeeze_b1(data_batch[self.config.text_embed_key]).to(**self.tensor_kwargs)
        ref_latent = None
        if self.config.ref_latent_key in data_batch:
            ref_latent = _squeeze_b1(data_batch[self.config.ref_latent_key]).to(**self.tensor_kwargs)
        return IT2AVBatch(v_x0=v_x0, a_x0=a_x0, text_emb=text_emb, ref_latent=ref_latent)

    def _sample_time(self, batch_size: int, device: torch.device) -> Tensor:
        # Uniform time in (0, 1). Keep away from exact 0/1 for numerical stability.
        t = torch.rand(batch_size, device=device, dtype=torch.float32)
        eps = 1e-5
        return t.clamp(min=eps, max=1.0 - eps)

    def _make_xt(self, x0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        # Rectified flow interpolation: x_t = (1-t) * x0 + t * eps
        eps_noise = torch.randn_like(x0, dtype=torch.float32, device=x0.device)
        x0_f = x0.float()
        if x0.ndim == 5:
            t_view = t.view(-1, 1, 1, 1, 1)
        elif x0.ndim == 3:
            t_view = t.view(-1, 1, 1)
        else:
            raise ValueError(f"Unexpected x0 ndim={x0.ndim}, expected 5 (video) or 3 (audio)")
        xt = (1.0 - t_view) * x0_f + t_view * eps_noise
        return xt.to(dtype=x0.dtype), eps_noise.to(dtype=x0.dtype)

    def _forward_video(self, net: torch.nn.Module, xt: Tensor, t: Tensor, text_emb: Tensor) -> Tensor:
        timestep = (t * self.config.rectified_flow_t_scaling_factor).to(dtype=torch.float32)
        amp_ctx = (
            torch.autocast("cuda", dtype=self.precision)
            if self.precision in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        with amp_ctx:
            return net(x=xt, timestep=timestep, context=text_emb)

    def _forward_audio(self, net: torch.nn.Module, xt: Tensor, t: Tensor, text_emb: Tensor) -> Tensor:
        timestep = (t * self.config.rectified_flow_t_scaling_factor).to(dtype=torch.float32)
        amp_ctx = (
            torch.autocast("cuda", dtype=self.precision)
            if self.precision in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        with amp_ctx:
            return net(x=xt, timestep=timestep, context=text_emb)

    def training_step(self, data_batch: Dict[str, Tensor], iteration: int = 0) -> Tuple[Dict[str, Tensor], Tensor]:
        batch = self._parse_batch(data_batch)

        bsz = batch.v_x0.shape[0]
        t = self._sample_time(bsz, device=batch.v_x0.device)

        v_xt, _ = self._make_xt(batch.v_x0, t)
        a_xt, _ = self._make_xt(batch.a_x0, t)

        v_pred = self._forward_video(self.student_video, v_xt, t, batch.text_emb).float()
        a_pred = self._forward_audio(self.student_audio, a_xt, t, batch.text_emb).float()

        with torch.no_grad():
            v_teacher = self._forward_video(self.teacher_video, v_xt, t, batch.text_emb).float()
            a_teacher = self._forward_audio(self.teacher_audio, a_xt, t, batch.text_emb).float()

        loss_v = torch.mean((v_pred - v_teacher) ** 2)
        loss_a = torch.mean((a_pred - a_teacher) ** 2)
        loss = loss_v + loss_a

        out = {
            "t": t.detach(),
            "v_xt": v_xt.detach(),
            "a_xt": a_xt.detach(),
            "v_teacher": v_teacher.detach(),
            "a_teacher": a_teacher.detach(),
            "v_pred": v_pred.detach(),
            "a_pred": a_pred.detach(),
            "loss_v": loss_v.detach(),
            "loss_a": loss_a.detach(),
        }
        return out, loss

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Use training_step() for distillation training.")

    def model_param_stats(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        learnable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_param_num": total, "total_learnable_param_num": learnable}

    def is_image_batch(self, data_batch: Dict[str, Tensor]) -> bool:
        return False

    def init_optimizer_scheduler(self, optimizer_config, scheduler_config):
        optimizer = lazy_instantiate(optimizer_config, model=self)
        scheduler = get_base_scheduler(optimizer, self, scheduler_config)
        return optimizer, scheduler

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        # Move teacher/student to GPU after trainer sets up distributed environment.
        self.teacher_video = self.teacher_video.to(memory_format=memory_format, **self.tensor_kwargs)
        self.teacher_audio = self.teacher_audio.to(memory_format=memory_format, **self.tensor_kwargs)
        self.student_video = self.student_video.to(memory_format=memory_format, **self.tensor_kwargs)
        self.student_audio = self.student_audio.to(memory_format=memory_format, **self.tensor_kwargs)

        # DDP requires all ranks to have the same parameters present; keep teacher as buffers-only (no grads).
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

