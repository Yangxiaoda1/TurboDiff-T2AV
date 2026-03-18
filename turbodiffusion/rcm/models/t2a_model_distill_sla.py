from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import attrs
import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from imaginaire.lazy_config import instantiate as lazy_instantiate
from imaginaire.model import ImaginaireModel
from imaginaire.utils import log
from rcm.utils.checkpointer import non_strict_load_model
from rcm.utils.optim_instantiate_dtensor import get_base_scheduler


class _IdentityTokenizer(torch.nn.Module):
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @torch.no_grad()
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _maybe_add_mova_to_syspath() -> None:
    mova_root = os.environ.get("MOVA_ROOT", "")
    if mova_root and os.path.isdir(mova_root):
        if mova_root not in sys.path:
            sys.path.append(mova_root)
        return

    here = os.path.dirname(os.path.abspath(__file__))
    turbo_root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
    candidate = os.path.join(os.path.dirname(turbo_root), "MOVA")
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.append(candidate)


def _truncate_blocks(module: torch.nn.Module, num_layers: int) -> None:
    if num_layers <= 0 or not hasattr(module, "blocks"):
        return
    blocks = getattr(module, "blocks")
    if not isinstance(blocks, torch.nn.ModuleList) or num_layers >= len(blocks):
        return
    setattr(module, "blocks", torch.nn.ModuleList(list(blocks)[:num_layers]))


@attrs.define(slots=False)
class T2ADistillConfig_SLA:
    conditioner: Any = None
    tokenizer: Any = None
    ema: Any = None

    a_latent_key: str = "a_latents"
    text_embed_key: str = "t5_text_embeddings"
    prompt_key: str = "prompts"

    mova_ckpt_path: str = ""
    precision: str = "bfloat16"
    rectified_flow_t_scaling_factor: float = 1000.0

    student_num_layers_audio: int = 0
    fd_size: float = 1e-3
    tangent_warmup: int = 1000
    loss_scale: float = 100.0
    teacher_guidance: float = 0.0


@dataclass
class T2ABatch:
    a_x0: Tensor
    text_emb: Tensor


class T2ADistillModel_SLA(ImaginaireModel):
    """Audio-only distillation using the MOVA audio DiT as teacher and student."""

    def __init__(self, config: T2ADistillConfig_SLA):
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
        except Exception as exc:
            raise ImportError(
                "Failed to import MOVA. Set MOVA_ROOT to your MOVA project directory. "
                f"Original error: {exc!r}"
            ) from exc

        if not self.config.mova_ckpt_path:
            raise ValueError("config.mova_ckpt_path must be set to the MOVA checkpoint directory.")

        log.info(f"Loading MOVA audio teacher from {self.config.mova_ckpt_path}")
        pipe = MOVA.from_pretrained(self.config.mova_ckpt_path, torch_dtype=self.precision)
        self.teacher_audio = pipe.audio_dit.eval().requires_grad_(False)
        self.student_audio = deepcopy(self.teacher_audio).train().requires_grad_(True)
        _truncate_blocks(self.student_audio, self.config.student_num_layers_audio)

        # Keep a T2V-compatible namespace so DCP can resolve nested keys like
        # "student.audio_dit.*" when building distributed state dicts.
        self.teacher = torch.nn.Module()
        self.teacher.audio_dit = self.teacher_audio
        self.student = torch.nn.Module()
        self.student.audio_dit = self.student_audio
        del pipe

        self.optimizer_dict: Dict[str, torch.optim.Optimizer] = {}
        self.scheduler_dict: Dict[str, torch.optim.lr_scheduler.LRScheduler] = {}

    def _parse_batch(self, data_batch: Dict[str, Any]) -> T2ABatch:
        def _squeeze_b1(x: Tensor) -> Tensor:
            if isinstance(x, torch.Tensor) and x.ndim >= 2 and x.shape[1] == 1:
                return x[:, 0]
            return x

        a_x0 = _squeeze_b1(data_batch[self.config.a_latent_key]).to(**self.tensor_kwargs)
        text_emb = _squeeze_b1(data_batch[self.config.text_embed_key]).to(**self.tensor_kwargs)
        return T2ABatch(a_x0=a_x0, text_emb=text_emb)

    def _sample_time(self, batch_size: int, device: torch.device) -> Tensor:
        h = float(self.config.fd_size)
        t = torch.rand(batch_size, device=device, dtype=torch.float32)
        return t.clamp(min=max(1e-5, h + 1e-5), max=1.0 - 1e-5)

    def _make_xt(self, x0: Tensor, theta: Tensor) -> Tensor:
        eps_noise = torch.randn_like(x0, dtype=torch.float32, device=x0.device)
        x0_f = x0.float()
        cos_view = torch.cos(theta).view(-1, 1, 1)
        sin_view = torch.sin(theta).view(-1, 1, 1)
        xt = cos_view * x0_f + sin_view * eps_noise
        return xt.to(dtype=x0.dtype)

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
        batch_size = batch.a_x0.shape[0]

        t = self._sample_time(batch_size, device=batch.a_x0.device)
        theta = 0.5 * torch.pi * t
        a_xt = self._make_xt(batch.a_x0, theta)

        h = float(self.config.fd_size)
        t_prev = (t - h).clamp(min=1e-5)
        theta_h = 0.5 * torch.pi * h
        cos_t = torch.cos(theta).view(-1, 1, 1)
        sin_t = torch.sin(theta).view(-1, 1, 1)

        with torch.no_grad():
            a_teacher = self._forward_audio(self.teacher_audio, a_xt, t, batch.text_emb).float()

        if self.config.teacher_guidance > 0.0:
            with torch.no_grad():
                uncond = torch.zeros_like(batch.text_emb)
                a_teacher_uncond = self._forward_audio(self.teacher_audio, a_xt, t, uncond).float()
                a_teacher = a_teacher + self.config.teacher_guidance * (a_teacher - a_teacher_uncond)

        a_pred = self._forward_audio(self.student_audio, a_xt, t, batch.text_emb).float()

        theta_h_tensor = torch.tensor(theta_h, device=a_xt.device, dtype=a_xt.dtype)
        with torch.no_grad():
            a_xt_prev = torch.cos(theta_h_tensor) * a_xt - torch.sin(theta_h_tensor) * a_teacher.to(dtype=a_xt.dtype)
            a_pred_prev = self._forward_audio(self.student_audio, a_xt_prev, t_prev, batch.text_emb).float()

        sin_h = max(float(torch.sin(torch.tensor(theta_h)).item()), 1e-6)
        cos_h = float(torch.cos(torch.tensor(theta_h)).item())
        dF_dt = (cos_h * a_pred - a_pred_prev) / sin_h

        a_pred_sg = a_pred.detach()
        warmup_ratio = min(1.0, float(iteration) / max(1, int(self.config.tangent_warmup)))
        g = -cos_t * (a_pred_sg - a_teacher) - warmup_ratio * (sin_t * a_xt.float() + cos_t * sin_t * dF_dt)

        nan_mask = torch.isnan(g).flatten(start_dim=1).any(dim=1).view(-1, 1, 1)
        g = torch.where(nan_mask, torch.zeros_like(g), g)
        a_pred = torch.where(nan_mask, torch.zeros_like(a_pred), a_pred)
        a_pred_sg = torch.where(nan_mask, torch.zeros_like(a_pred_sg), a_pred_sg)

        g_norm = g.double().flatten(start_dim=1).norm(p=2, dim=1, keepdim=True).view(-1, 1, 1)
        g = (g.double() / (g_norm + 0.1)).to(dtype=a_pred.dtype)

        loss_scm = ((a_pred - a_pred_sg - g) ** 2).flatten(start_dim=1).sum(dim=1)
        loss = self.config.loss_scale * loss_scm.mean()

        # 打印 student_audio 第一层权重的梯度
        for name, param in self.student_audio.named_parameters():
            if 'weight' in name:
                if param.grad is not None:
                    print(f'[GradCheck] iter={iteration} {name} grad mean={param.grad.mean().item()} grad std={param.grad.std().item()}')
                else:
                    print(f'[GradCheck] iter={iteration} {name} grad=None')
                break

        output = {
            "t": t.detach(),
            "a_xt": a_xt.detach(),
            "a_teacher": a_teacher.detach(),
            "a_pred": a_pred.detach(),
            "a_pred_prev": a_pred_prev.detach(),
            "df_dt": dF_dt.detach(),
            "loss_a": loss.detach(),
        }
        return output, loss

    @torch.no_grad()
    def validation_step(self, data_batch: Dict[str, Tensor], iteration: int = 0) -> Tuple[Dict[str, Tensor], Tensor]:
        return self.training_step(data_batch, iteration)

    def forward(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("Use training_step() for distillation training.")

    def model_param_stats(self) -> Dict[str, int]:
        total = sum(param.numel() for param in self.parameters())
        learnable = sum(param.numel() for param in self.parameters() if param.requires_grad)
        return {"total_param_num": total, "total_learnable_param_num": learnable}

    def model_dict(self) -> Dict[str, Any]:
        return {"net": self}

    def is_image_batch(self, data_batch: Dict[str, Tensor]) -> bool:
        return False

    def init_optimizer_scheduler(self, optimizer_config, scheduler_config):
        optimizer = lazy_instantiate(optimizer_config, model=self)
        scheduler = get_base_scheduler(optimizer, self, scheduler_config)
        self.optimizer_dict = {"net": optimizer}
        self.scheduler_dict = {"net": scheduler}
        return optimizer, scheduler

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        self.teacher_audio = self.teacher_audio.to(memory_format=memory_format, **self.tensor_kwargs)
        self.student_audio = self.student_audio.to(memory_format=memory_format, **self.tensor_kwargs)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def clip_grad_norm_(
        self,
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: Optional[bool] = None,
    ):
        for param in self.student_audio.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0, out=param.grad)
        return clip_grad_norm_(
            self.student_audio.parameters(),
            max_norm=max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        ).cpu()

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.student_audio.state_dict(prefix="student.audio_dit.")

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        student_state = {
            key.replace("student.audio_dit.", ""): value
            for key, value in state_dict.items()
            if key.startswith("student.audio_dit.")
        }
        if strict:
            return self.student_audio.load_state_dict(student_state, strict=True, assign=assign)
        log.critical("load T2A audio student in non-strict mode")
        return non_strict_load_model(self.student_audio, dict(student_state))
