from __future__ import annotations

import gc
import math
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

from imaginaire.callbacks.low_precision import update_master_weights
from imaginaire.lazy_config import instantiate as lazy_instantiate
from imaginaire.model import ImaginaireModel
from imaginaire.utils import log
from rcm.utils.checkpointer import non_strict_load_model
from rcm.utils.denoiser_scaling import RectifiedFlow_TrigFlowWrapper
from rcm.utils.lognormal import LogNormal
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


def _extract_audio_state_from_flat_dict(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    audio_prefixes = ("audio.", "audio_dit.", "student.audio_dit.", "student.audio.")
    audio_state = {}
    for key, value in state_dict.items():
        for prefix in audio_prefixes:
            if key.startswith(prefix):
                audio_state[key[len(prefix) :]] = value
                break
    return audio_state


def _is_distill_checkpoint_path(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    if os.path.isdir(os.path.join(path, "model")):
        return True
    return any(name.endswith('.distcp') for name in os.listdir(path))


def _load_audio_state_from_pt(student_ckpt_path: str) -> Dict[str, Any]:
    state_dict = torch.load(student_ckpt_path, map_location='cpu')
    if not isinstance(state_dict, dict):
        raise ValueError('student_ckpt_path must point to a dict-like checkpoint')
    audio_state = _extract_audio_state_from_flat_dict(state_dict)
    if not audio_state:
        raise ValueError(
            'No audio branch weights found in student checkpoint. '
            'Expected prefixes: audio.*, audio_dit.*, student.audio_dit.*'
        )
    return audio_state


def _load_audio_state_from_dcp(audio_dit: torch.nn.Module, dcp_dir: str) -> Dict[str, Any]:
    try:
        import torch.distributed.checkpoint as dcp
    except Exception as exc:
        raise RuntimeError(f'Failed to import torch.distributed.checkpoint: {exc}') from exc

    dcp_path = dcp_dir
    if os.path.isdir(os.path.join(dcp_dir, 'model')):
        dcp_path = os.path.join(dcp_dir, 'model')

    request_state = {}
    for key, value in audio_dit.state_dict().items():
        request_state[f'student.audio_dit.{key}'] = torch.empty_like(value, device='cpu')

    dcp.load(state_dict=request_state, checkpoint_id=dcp_path)

    audio_state = {}
    prefix = 'student.audio_dit.'
    for key, value in request_state.items():
        if key.startswith(prefix):
            audio_state[key[len(prefix) :]] = value
    return audio_state


def _load_audio_branch_weights(audio_dit: torch.nn.Module, student_ckpt_path: str) -> None:
    if os.path.isdir(student_ckpt_path):
        audio_state = _load_audio_state_from_dcp(audio_dit, student_ckpt_path)
        src_type = 'DCP'
    else:
        audio_state = _load_audio_state_from_pt(student_ckpt_path)
        src_type = 'PT'

    missing, unexpected = audio_dit.load_state_dict(audio_state, strict=False)
    log.info(
        f'Loaded student audio weights ({src_type}): {len(audio_state)} tensors, '
        f'missing={len(missing)}, unexpected={len(unexpected)}'
    )


@attrs.define(slots=False)
class T2ADistillConfig_rCM:
    conditioner: Any = None
    tokenizer: Any = None
    ema: Any = None

    a_latent_key: str = 'a_latents'
    text_embed_key: str = 't5_text_embeddings'
    prompt_key: str = 'prompts'

    mova_ckpt_path: str = ''
    teacher_ckpt_path: str = ''
    student_ckpt_path: str = ''

    precision: str = 'bfloat16'
    rectified_flow_t_scaling_factor: float = 1000.0
    sigma_data: float = 1.0

    student_num_layers_audio: int = 0
    iteration_offset: int = 0
    student_update_freq: int = 5
    max_simulation_steps_fake: int = 4

    loss_scale: float = 100.0
    loss_scale_dmd: float = 1.0
    loss_scale_fake_score: float = 1.0
    loss_scale_teacher: float = 0.0
    tangent_warmup: int = 1000
    teacher_guidance: float = 5.0
    negative_prompt: str = 'noise, distorted, low quality, bad audio, muffled, static'

    p_mean: float = -0.8
    p_std: float = 1.6
    p_D_mean: float = 0.0
    p_D_std: float = 1.6
    timestep_shift: float = 5.0

    fake_score_lr: float = 4e-7
    fake_score_weight_decay: float = 0.01
    fake_score_betas: Tuple[float, float] = (0.0, 0.999)

    fd_type: int = 0
    fd_size: float = 1e-4


@dataclass
class T2ABatch:
    a_x0: Tensor
    text_emb: Tensor


class T2ADistillModel_rCM(ImaginaireModel):
    """Audio-only rCM distillation using MOVA audio teacher/student/fake-score branches."""

    @staticmethod
    def _compute_t5_embeds(text_encoder, tokenizer, prompt: str, device: str = 'cpu') -> torch.Tensor:
        text_inputs = tokenizer(
            [prompt],
            padding='max_length',
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        with torch.no_grad():
            embeds = text_encoder(
                text_inputs.input_ids.to(device),
                text_inputs.attention_mask.to(device),
            ).last_hidden_state

        seq_lens = text_inputs.attention_mask.gt(0).sum(dim=1)
        embeds = [u[:v] for u, v in zip(embeds, seq_lens)]
        return torch.stack(
            [torch.cat([u, u.new_zeros(512 - u.size(0), u.size(1))]) for u in embeds],
            dim=0,
        )

    def __init__(self, config: T2ADistillConfig_rCM):
        super().__init__()
        self.config = config
        self.precision = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }[config.precision]
        self.tensor_kwargs = {'device': 'cuda', 'dtype': self.precision}
        self.tokenizer = _IdentityTokenizer()
        self.scaling = RectifiedFlow_TrigFlowWrapper(
            sigma_data=config.sigma_data,
            t_scaling_factor=config.rectified_flow_t_scaling_factor,
        )
        self.p_G = LogNormal(p_mean=config.p_mean, p_std=config.p_std)
        self.p_D = LogNormal(p_mean=config.p_D_mean, p_std=config.p_D_std)

        _maybe_add_mova_to_syspath()
        try:
            from mova.diffusion.models import WanAudioModel  # type: ignore
        except Exception as exc:
            raise ImportError(
                'Failed to import MOVA audio model. Set MOVA_ROOT to your MOVA project directory. '
                f'Original error: {exc!r}'
            ) from exc

        teacher_ckpt_path = config.teacher_ckpt_path or config.mova_ckpt_path
        student_ckpt_path = config.student_ckpt_path or teacher_ckpt_path
        if not teacher_ckpt_path:
            raise ValueError('config.teacher_ckpt_path or config.mova_ckpt_path must be set.')

        log.info(f'Loading MOVA audio teacher from {teacher_ckpt_path}')
        self.teacher_audio = WanAudioModel.from_pretrained(
            teacher_ckpt_path,
            subfolder='audio_dit',
            torch_dtype=self.precision,
        ).eval().requires_grad_(False)

        self.student_audio = deepcopy(self.teacher_audio).train().requires_grad_(True)
        if student_ckpt_path and student_ckpt_path != teacher_ckpt_path:
            if _is_distill_checkpoint_path(student_ckpt_path) or os.path.isfile(student_ckpt_path):
                log.info(f'Loading distilled student audio weights from {student_ckpt_path}')
                _load_audio_branch_weights(self.student_audio, student_ckpt_path)
            else:
                log.info(f'Loading MOVA audio student from {student_ckpt_path}')
                self.student_audio = WanAudioModel.from_pretrained(
                    student_ckpt_path,
                    subfolder='audio_dit',
                    torch_dtype=self.precision,
                ).train().requires_grad_(True)

        _truncate_blocks(self.student_audio, self.config.student_num_layers_audio)

        self.fake_score_audio: Optional[torch.nn.Module]
        if self.config.loss_scale_dmd > 0.0 and self.config.loss_scale_fake_score > 0.0:
            self.fake_score_audio = deepcopy(self.teacher_audio).train().requires_grad_(True)
        else:
            self.fake_score_audio = None

        self.teacher = torch.nn.Module()
        self.teacher.audio_dit = self.teacher_audio
        self.student = torch.nn.Module()
        self.student.audio_dit = self.student_audio
        self.fake_score = None
        if self.fake_score_audio is not None:
            self.fake_score = torch.nn.Module()
            self.fake_score.audio_dit = self.fake_score_audio

        self.neg_text_emb: Optional[Tensor] = None
        if float(self.config.teacher_guidance) > 1.0:
            neg_prompt = str(getattr(self.config, 'negative_prompt', '')).strip()
            if not neg_prompt:
                raise ValueError('CFG enabled but config.negative_prompt is empty.')
            log.info('Loading T5 for T2A CFG negative prompt embedding...')
            try:
                from transformers import T5TokenizerFast, UMT5EncoderModel
            except Exception as exc:
                raise ImportError(
                    'CFG enabled for T2A but transformers T5 components could not be imported. '
                    f'Original error: {exc!r}'
                ) from exc

            text_dtype = self.precision if self.precision != torch.float16 else torch.float32
            tokenizer = T5TokenizerFast.from_pretrained(teacher_ckpt_path, subfolder='tokenizer')
            text_encoder = UMT5EncoderModel.from_pretrained(
                teacher_ckpt_path,
                subfolder='text_encoder',
                torch_dtype=text_dtype,
            )
            self.neg_text_emb = self._compute_t5_embeds(text_encoder, tokenizer, neg_prompt, device='cpu').to(
                dtype=self.precision,
                device='cpu',
            )
            del tokenizer
            del text_encoder
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log.info('Prepared T2A negative prompt embedding for teacher CFG.')

        self.optimizer_dict: Dict[str, torch.optim.Optimizer] = {}
        self.scheduler_dict: Dict[str, torch.optim.lr_scheduler.LRScheduler] = {}
        self._debug_last_phase = 'student phase'
        self._debug_last_losses: Dict[str, float] = {
            'scm_loss': 0.0,
            'dmd_loss': 0.0,
            'fake_score_loss': 0.0,
        }

    def _effective_iteration(self, iteration: int) -> int:
        return int(iteration) + int(self.config.iteration_offset)

    def _parse_batch(self, data_batch: Dict[str, Any]) -> T2ABatch:
        def _squeeze_b1(x: Tensor) -> Tensor:
            if isinstance(x, torch.Tensor) and x.ndim >= 2 and x.shape[1] == 1:
                return x[:, 0]
            return x

        a_x0 = _squeeze_b1(data_batch[self.config.a_latent_key]).to(**self.tensor_kwargs)
        text_emb = _squeeze_b1(data_batch[self.config.text_embed_key]).to(**self.tensor_kwargs)
        return T2ABatch(a_x0=a_x0, text_emb=text_emb)

    def _broadcast_time(self, time_B: Tensor, ndim: int) -> Tensor:
        if ndim == 3:
            return time_B.view(-1, 1, 1)
        raise ValueError(f'Unsupported ndim={ndim}')

    def _draw_time_G(self, batch_size: int) -> Tensor:
        sigma = self.p_G(batch_size)
        return torch.arctan(sigma).double()

    def _draw_time_D(self, batch_size: int) -> Tensor:
        if self.config.timestep_shift > 0:
            sigma = torch.rand(batch_size, device='cuda', dtype=torch.float64)
            sigma = self.config.timestep_shift * sigma / (1 + (self.config.timestep_shift - 1) * sigma)
            return torch.arctan(sigma / torch.clamp(1 - sigma, min=1e-8))
        sigma = self.p_D(batch_size)
        return torch.arctan(sigma).double()

    def _get_uncond_text(self, text_emb: Tensor) -> Tensor:
        if self.neg_text_emb is not None:
            uncond = self.neg_text_emb.to(device=text_emb.device, dtype=text_emb.dtype)
            if uncond.shape[0] != text_emb.shape[0]:
                uncond = uncond.expand(text_emb.shape[0], -1, -1)
            return uncond
        return torch.zeros_like(text_emb)

    def _denoise_audio(
        self,
        a_xt: Tensor,
        time_B: Tensor,
        text_emb: Tensor,
        net: torch.nn.Module,
    ) -> Tuple[Tensor, Tensor]:
        a_tv = self._broadcast_time(time_B, a_xt.ndim)
        a_skip, a_out, a_in, a_noise = self.scaling(trigflow_t=a_tv)

        amp_ctx = (
            torch.autocast('cuda', dtype=self.precision)
            if self.precision in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        with amp_ctx:
            a_scaled = (a_xt * a_in).to(**self.tensor_kwargs)
            a_raw = net(
                x=a_scaled,
                timestep=a_noise.reshape(-1).to(device=a_xt.device, dtype=torch.float32),
                context=text_emb,
            )

        a_x0 = (a_skip * a_xt + a_out * a_raw).to(dtype=a_xt.dtype)
        ac, a_s = torch.cos(a_tv), torch.sin(a_tv)
        a_F = (ac * a_xt - a_x0) / a_s
        return a_x0, a_F

    def is_student_phase(self, iteration: int) -> bool:
        effective_iter = self._effective_iteration(iteration)
        return (
            self.fake_score_audio is None
            or effective_iter < self.config.tangent_warmup
            or effective_iter % max(1, self.config.student_update_freq) == 0
        )

    def get_effective_iteration(self, iteration: int) -> int:
        effective_iter = self._effective_iteration(iteration)
        if self.fake_score_audio is None or effective_iter < self.config.tangent_warmup:
            return effective_iter
        return self.config.tangent_warmup + (effective_iter - self.config.tangent_warmup) // max(1, self.config.student_update_freq)

    def get_effective_iteration_fake(self, iteration: int) -> int:
        return self._effective_iteration(iteration) - self.get_effective_iteration(iteration) - 1

    def training_step_generator(self, batch: T2ABatch, iteration: int) -> Tuple[Dict[str, Tensor], Tensor]:
        bsz = batch.a_x0.shape[0]
        effective_iter = self._effective_iteration(iteration)

        time_B = self._draw_time_G(bsz)
        a_tv = self._broadcast_time(time_B, batch.a_x0.ndim)
        ac, a_s = torch.cos(a_tv), torch.sin(a_tv)
        a_xt = (batch.a_x0.float() * ac + torch.randn_like(batch.a_x0.float()) * a_s).to(batch.a_x0.dtype)
        uncond = self._get_uncond_text(batch.text_emb)

        with torch.no_grad():
            _, a_F_teacher = self._denoise_audio(a_xt, time_B, batch.text_emb, self.teacher_audio)
            if self.config.teacher_guidance > 0.0:
                _, a_F_uncond = self._denoise_audio(a_xt, time_B, uncond, self.teacher_audio)
                a_F_teacher = a_F_teacher + self.config.teacher_guidance * (a_F_teacher - a_F_uncond)

        G_x0_theta: Optional[Tensor] = None
        if self.fake_score_audio is not None and effective_iter > self.config.tangent_warmup:
            G_time_B = torch.full_like(time_B, math.pi / 2)
            G_xt = torch.randn_like(batch.a_x0.float()).to(batch.a_x0.dtype)
            num_simulation_steps_fake = self.get_effective_iteration(iteration) % max(1, self.config.max_simulation_steps_fake)
            for _ in range(num_simulation_steps_fake):
                with torch.no_grad():
                    G_x0, _ = self._denoise_audio(G_xt, G_time_B, batch.text_emb, self.student_audio)
                G_time_B = torch.minimum(self._draw_time_D(bsz), G_time_B)
                G_tv = self._broadcast_time(G_time_B, batch.a_x0.ndim)
                G_xt = (G_x0.float() * torch.cos(G_tv) + torch.randn_like(batch.a_x0.float()) * torch.sin(G_tv)).to(batch.a_x0.dtype)

            all_xt = torch.cat([a_xt, G_xt], dim=0)
            all_time_B = torch.cat([time_B, G_time_B], dim=0)
            all_text_emb = torch.cat([batch.text_emb, batch.text_emb], dim=0)
            all_x0_pred, all_F = self._denoise_audio(all_xt, all_time_B, all_text_emb, self.student_audio)
            a_x0_pred, G_x0_theta = torch.chunk(all_x0_pred, 2, dim=0)
            a_F, _ = torch.chunk(all_F, 2, dim=0)
        else:
            a_x0_pred, a_F = self._denoise_audio(a_xt, time_B, batch.text_emb, self.student_audio)

        a_F_sg = a_F.detach()
        h = float(self.config.fd_size)
        time_lo = 1e-5
        time_hi = math.pi / 2 - 1e-5

        if self.config.fd_type == 0:
            # JVP-like central finite difference along the teacher tangent direction.
            t_xt = (ac * a_s * a_F_teacher).detach()
            t_time = (ac * a_s).reshape(-1).detach()
            time_plus = (time_B + h * t_time).clamp(min=time_lo, max=time_hi)
            time_minus = (time_B - h * t_time).clamp(min=time_lo, max=time_hi)
            a_xt_plus = (a_xt.float() + h * t_xt.float()).to(dtype=a_xt.dtype)
            a_xt_minus = (a_xt.float() - h * t_xt.float()).to(dtype=a_xt.dtype)
            with torch.no_grad():
                _, a_F_plus = self._denoise_audio(a_xt_plus, time_plus, batch.text_emb, self.student_audio)
                _, a_F_minus = self._denoise_audio(a_xt_minus, time_minus, batch.text_emb, self.student_audio)
            dF_dt = (a_F_plus - a_F_minus) / max(2.0 * h, 1e-8)
        elif self.config.fd_type == 1:
            # Semi-discrete difference: keep x_t fixed and move only in time.
            time_prev = (time_B - h).clamp(min=time_lo, max=time_hi)
            with torch.no_grad():
                _, a_F_prev = self._denoise_audio(a_xt, time_prev, batch.text_emb, self.student_audio)
            dF_dt = (math.cos(h) * a_F.detach() - a_F_prev) / max(math.sin(h), 1e-8)
        elif self.config.fd_type == 2:
            # Discrete difference: move x_t along teacher flow to t-h, then finite diff.
            time_prev = (time_B - h).clamp(min=time_lo, max=time_hi)
            a_xt_prev = math.cos(h) * a_xt - math.sin(h) * a_F_teacher
            with torch.no_grad():
                _, a_F_prev = self._denoise_audio(a_xt_prev, time_prev, batch.text_emb, self.student_audio)
            dF_dt = (math.cos(h) * a_F.detach() - a_F_prev) / max(math.sin(h), 1e-8)
        else:
            raise NotImplementedError(f'Unsupported fd_type={self.config.fd_type}. Expected one of [0, 1, 2].')

        warmup_ratio = min(1.0, float(effective_iter) / max(1.0, float(self.config.tangent_warmup)))
        tangent = ac * a_s * dF_dt
        g = -ac * torch.sqrt(torch.clamp(1 - warmup_ratio**2 * a_s**2, min=0.0)) * (a_F_sg - a_F_teacher)
        g = g - warmup_ratio * (ac * a_s * a_xt + tangent)

        with torch.no_grad():
            df_dt = -ac * (a_F_sg - a_F_teacher) - (a_s * a_xt + tangent)
            sample_nan_mask = (
                torch.isnan(g).flatten(start_dim=1).any(dim=1)
                | torch.isnan(a_F).flatten(start_dim=1).any(dim=1)
            ).view(-1, 1, 1)
            nan_mask = sample_nan_mask.expand_as(g)

        g = torch.where(nan_mask, torch.zeros_like(g), g)
        a_F = torch.where(nan_mask, torch.zeros_like(a_F), a_F)
        a_F_sg = torch.where(nan_mask, torch.zeros_like(a_F_sg), a_F_sg)
        g = g.double() / (g.double().norm(p=2, dim=tuple(range(1, g.ndim)), keepdim=True) + 0.1)
        g = g.to(dtype=a_F.dtype)

        loss_scm = ((a_F - a_F_sg - g) ** 2).sum(dim=tuple(range(1, a_F.ndim)))
        loss = self.config.loss_scale * loss_scm.mean()
        x0_teacher = (ac * a_xt - a_s * a_F_teacher).to(dtype=a_x0_pred.dtype)
        loss_teacher_value = torch.zeros((), device=loss.device, dtype=loss.dtype)
        if float(self.config.loss_scale_teacher) > 0.0:
            loss_teacher = ((a_x0_pred - x0_teacher.detach()) ** 2).sum(dim=tuple(range(1, a_x0_pred.ndim)))
            loss_teacher_value = self.config.loss_scale_teacher * loss_teacher.mean().to(dtype=loss.dtype)
            loss = loss + loss_teacher_value
        loss_dmd_value = torch.zeros((), device=loss.device, dtype=loss.dtype)

        if self.fake_score_audio is not None and effective_iter > self.config.tangent_warmup and G_x0_theta is not None:
            D_time_B = self._draw_time_D(bsz)
            D_tv = self._broadcast_time(D_time_B, batch.a_x0.ndim)
            D_xt_theta = (G_x0_theta.float() * torch.cos(D_tv) + torch.randn_like(batch.a_x0.float()) * torch.sin(D_tv)).to(batch.a_x0.dtype)

            with torch.no_grad():
                x0_theta_fake, _ = self._denoise_audio(D_xt_theta, D_time_B, batch.text_emb, self.fake_score_audio)
                x0_theta_teacher, _ = self._denoise_audio(D_xt_theta, D_time_B, batch.text_emb, self.teacher_audio)
                if self.config.teacher_guidance > 0.0:
                    x0_theta_teacher_uncond, _ = self._denoise_audio(D_xt_theta, D_time_B, uncond, self.teacher_audio)
                    x0_theta_teacher = x0_theta_teacher + self.config.teacher_guidance * (
                        x0_theta_teacher - x0_theta_teacher_uncond
                    )
                weight_factor = (
                    torch.abs(G_x0_theta.double() - x0_theta_teacher.double())
                    .mean(dim=tuple(range(1, G_x0_theta.ndim)), keepdim=True)
                    .clip(min=1e-5)
                )
            grad = (x0_theta_fake.double() - x0_theta_teacher.double()) / weight_factor
            loss_dmd = (G_x0_theta.double() - (G_x0_theta.double() - grad).detach()) ** 2
            dmd_nan_mask = torch.isnan(loss_dmd).flatten(start_dim=1).any(dim=1).view(-1, 1, 1)
            loss_dmd = torch.where(dmd_nan_mask.expand_as(loss_dmd), torch.zeros_like(loss_dmd), loss_dmd)
            loss_dmd = loss_dmd.sum(dim=tuple(range(1, loss_dmd.ndim)))
            loss_dmd_value = self.config.loss_scale_dmd * loss_dmd.mean().to(dtype=loss.dtype)
            loss = loss + loss_dmd_value

        output = {
            'time': time_B.detach(),
            'a_xt': a_xt.detach(),
            'a_x0': batch.a_x0.detach(),
            'a_x0_pred': a_x0_pred.detach(),
            'a_F_teacher': a_F_teacher.detach(),
            'a_F_pred': a_F.detach(),
            'df_dt': df_dt.detach(),
            'loss_a': loss.detach(),
            'loss_scm': (self.config.loss_scale * loss_scm.mean()).detach(),
            'loss_teacher': loss_teacher_value.detach(),
            'loss_dmd': loss_dmd_value.detach(),
        }
        self._debug_last_phase = 'student phase'
        self._debug_last_losses = {
            'scm_loss': float((self.config.loss_scale * loss_scm.mean()).detach().item()),
            'dmd_loss': float(loss_dmd_value.detach().item()),
            'fake_score_loss': 0.0,
        }
        return output, loss

    def training_step_critic(self, batch: T2ABatch, iteration: int) -> Tuple[Dict[str, Tensor], Tensor]:
        if self.fake_score_audio is None:
            raise RuntimeError('Critic phase requested without fake-score branch.')

        bsz = batch.a_x0.shape[0]
        G_time_B = torch.full((bsz,), math.pi / 2, device=batch.a_x0.device, dtype=torch.float64)
        G_xt = torch.randn_like(batch.a_x0.float()).to(batch.a_x0.dtype)

        num_simulation_steps_fake = self.get_effective_iteration_fake(iteration) % max(1, self.config.max_simulation_steps_fake)
        for _ in range(num_simulation_steps_fake):
            with torch.no_grad():
                G_x0, _ = self._denoise_audio(G_xt, G_time_B, batch.text_emb, self.student_audio)
            G_time_B = torch.minimum(self._draw_time_D(bsz), G_time_B)
            G_tv = self._broadcast_time(G_time_B, batch.a_x0.ndim)
            G_xt = (G_x0.float() * torch.cos(G_tv) + torch.randn_like(batch.a_x0.float()) * torch.sin(G_tv)).to(batch.a_x0.dtype)

        with torch.no_grad():
            G_x0_theta, _ = self._denoise_audio(G_xt, G_time_B, batch.text_emb, self.student_audio)

        D_time_B = self._draw_time_D(bsz)
        D_tv = self._broadcast_time(D_time_B, batch.a_x0.ndim)
        D_xt_theta = (G_x0_theta.float() * torch.cos(D_tv) + torch.randn_like(batch.a_x0.float()) * torch.sin(D_tv)).to(batch.a_x0.dtype)
        x0_theta_fake, _ = self._denoise_audio(D_xt_theta, D_time_B, batch.text_emb, self.fake_score_audio)
        denom = torch.clamp(torch.sin(D_tv) ** 2, min=1e-6)
        loss = self.config.loss_scale_fake_score * (((G_x0_theta - x0_theta_fake) ** 2) / denom).sum(dim=tuple(range(1, G_x0_theta.ndim))).mean()

        output = {
            'time': D_time_B.detach(),
            'a_xt': G_xt.detach(),
            'a_x0_pred': G_x0_theta.detach(),
            'a_fake_score_pred': x0_theta_fake.detach(),
            'loss_a': loss.detach(),
            'loss_fake_score': loss.detach(),
        }
        self._debug_last_phase = 'fake_score phase'
        self._debug_last_losses = {
            'scm_loss': 0.0,
            'dmd_loss': 0.0,
            'fake_score_loss': float(loss.detach().item()),
        }
        return output, loss

    def training_step(self, data_batch: Dict[str, Tensor], iteration: int = 0) -> Tuple[Dict[str, Tensor], Tensor]:
        batch = self._parse_batch(data_batch)
        if self.is_student_phase(iteration):
            self.student_audio.train().requires_grad_(True)
            if self.fake_score_audio is not None:
                self.fake_score_audio.eval().requires_grad_(False)
            return self.training_step_generator(batch, iteration)

        self.student_audio.eval().requires_grad_(False)
        self.fake_score_audio.train().requires_grad_(True)
        return self.training_step_critic(batch, iteration)

    @torch.no_grad()
    def validation_step(self, data_batch: Dict[str, Tensor], iteration: int = 0) -> Tuple[Dict[str, Tensor], Tensor]:
        batch = self._parse_batch(data_batch)
        return self.training_step_generator(batch, iteration)

    def forward(self, *args: Any, **kwargs: Any):
        raise NotImplementedError('Use training_step() for distillation training.')

    def model_param_stats(self) -> Dict[str, int]:
        total = sum(param.numel() for param in self.parameters())
        learnable = sum(param.numel() for param in self.parameters() if param.requires_grad)
        return {'total_param_num': total, 'total_learnable_param_num': learnable}

    def model_dict(self) -> Dict[str, Any]:
        model_dict = {'net': self.student_audio}
        if self.fake_score_audio is not None:
            model_dict['fake_score'] = self.fake_score_audio
        return model_dict

    def is_image_batch(self, data_batch: Dict[str, Tensor]) -> bool:
        return False

    def init_optimizer_scheduler(self, optimizer_config, scheduler_config):
        optimizer = lazy_instantiate(optimizer_config, model=self.student_audio)
        scheduler = get_base_scheduler(optimizer, self, scheduler_config)
        self.optimizer_dict = {'net': optimizer}
        self.scheduler_dict = {'net': scheduler}

        if self.fake_score_audio is not None:
            fake_score_optimizer_config = deepcopy(optimizer_config)
            fake_score_optimizer_config['lr'] = self.config.fake_score_lr
            fake_score_optimizer_config['weight_decay'] = self.config.fake_score_weight_decay
            fake_score_optimizer_config['betas'] = list(self.config.fake_score_betas)
            fake_score_optimizer = lazy_instantiate(fake_score_optimizer_config, model=self.fake_score_audio)
            fake_score_scheduler = get_base_scheduler(fake_score_optimizer, self, scheduler_config)
            self.optimizer_dict['fake_score'] = fake_score_optimizer
            self.scheduler_dict['fake_score'] = fake_score_scheduler
        return optimizer, scheduler

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        if self.is_student_phase(iteration):
            return [self.optimizer_dict['net']]
        return [self.optimizer_dict['fake_score']]

    def get_lr_schedulers(self, iteration: int) -> list[torch.optim.lr_scheduler.LRScheduler]:
        if self.is_student_phase(iteration):
            return [self.scheduler_dict['net']]
        return [self.scheduler_dict['fake_score']]

    def optimizers_zero_grad(self, iteration: int) -> None:
        for optimizer in self.get_optimizers(iteration):
            optimizer.zero_grad(set_to_none=True)

    def optimizers_schedulers_step(self, grad_scaler: torch.cuda.amp.GradScaler, iteration: int) -> None:
        for optimizer in self.get_optimizers(iteration):
            grad_scaler.step(optimizer)
            grad_scaler.update()
        for scheduler in self.get_lr_schedulers(iteration):
            scheduler.step()

    def on_before_zero_grad(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        iteration: int,
    ) -> None:
        del optimizer, scheduler
        if not self.is_student_phase(iteration) and self.fake_score_audio is not None:
            update_master_weights(self.optimizer_dict['fake_score'])

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        self.teacher_audio = self.teacher_audio.to(memory_format=memory_format, **self.tensor_kwargs)
        self.student_audio = self.student_audio.to(memory_format=memory_format, **self.tensor_kwargs)
        if self.fake_score_audio is not None:
            self.fake_score_audio = self.fake_score_audio.to(memory_format=memory_format, **self.tensor_kwargs)
        if self.neg_text_emb is not None:
            self.neg_text_emb = self.neg_text_emb.to(**self.tensor_kwargs)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def clip_grad_norm_(
        self,
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: Optional[bool] = None,
    ):
        params = list(self.student_audio.parameters())
        if self.fake_score_audio is not None:
            params.extend(self.fake_score_audio.parameters())
        for param in params:
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0, out=param.grad)
        return clip_grad_norm_(
            params,
            max_norm=max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        ).cpu()

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        state_dict = self.student_audio.state_dict(prefix='student.audio_dit.')
        if self.fake_score_audio is not None:
            state_dict.update(self.fake_score_audio.state_dict(prefix='fake_score.audio_dit.'))
        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        student_state = {
            key.replace('student.audio_dit.', ''): value
            for key, value in state_dict.items()
            if key.startswith('student.audio_dit.')
        }
        fake_score_state = {
            key.replace('fake_score.audio_dit.', ''): value
            for key, value in state_dict.items()
            if key.startswith('fake_score.audio_dit.')
        }
        if strict:
            student_result = self.student_audio.load_state_dict(student_state, strict=True, assign=assign)
            if self.fake_score_audio is not None and fake_score_state:
                self.fake_score_audio.load_state_dict(fake_score_state, strict=True, assign=assign)
            elif self.fake_score_audio is not None and not fake_score_state:
                log.warning('Checkpoint does not contain fake_score weights; keeping current fake_score initialization.')
            return student_result

        log.critical('load T2A audio student/fake_score in non-strict mode')
        non_strict_load_model(self.student_audio, dict(student_state))
        if self.fake_score_audio is not None and fake_score_state:
            non_strict_load_model(self.fake_score_audio, dict(fake_score_state))
        return None
