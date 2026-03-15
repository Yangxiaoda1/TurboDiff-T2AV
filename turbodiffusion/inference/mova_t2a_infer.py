import argparse
import os
import sys
import time
import torch
import soundfile as sf

from transformers import T5TokenizerFast, UMT5EncoderModel


def _maybe_add_mova_to_syspath():
    candidates = []
    env_root = os.environ.get("MOVA_ROOT", "").strip()
    if env_root:
        candidates.append(env_root)
    candidates.extend(
        [
            "/home/lyx/turbodiffusion/MOVA",
            "/mnt/16T/home_user/lyx/turbodiffusion/MOVA",
        ]
    )
    for p in candidates:
        if p and os.path.isdir(p):
            if p not in sys.path:
                sys.path.insert(0, p)
            return p
    return None


def _extract_audio_state_from_flat_dict(state_dict):
    audio_prefixes = ("audio.", "audio_dit.", "student.audio_dit.", "student.audio.")
    audio_state = {}
    for k, v in state_dict.items():
        for prefix in audio_prefixes:
            if k.startswith(prefix):
                audio_state[k[len(prefix):]] = v
                break
    return audio_state


def _load_student_audio_weights_from_pt(audio_dit, student_ckpt_path):
    state_dict = torch.load(student_ckpt_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise ValueError("student_ckpt_path must point to a dict-like checkpoint")
    audio_state = _extract_audio_state_from_flat_dict(state_dict)
    if not audio_state:
        raise ValueError(
            "No audio branch weights found in student checkpoint. "
            "Expected prefixes: audio.*, audio_dit.*, student.audio_dit.*"
        )
    return audio_state


def _load_student_audio_weights_from_dcp(audio_dit, dcp_dir):
    try:
        import torch.distributed.checkpoint as dcp
    except Exception as e:
        raise RuntimeError(f"Failed to import torch.distributed.checkpoint: {e}") from e

    dcp_path = dcp_dir
    if os.path.isdir(os.path.join(dcp_dir, "model")):
        dcp_path = os.path.join(dcp_dir, "model")

    # Build expected DCP keys from current audio model structure.
    request_state = {}
    for k, v in audio_dit.state_dict().items():
        request_state[f"student.audio_dit.{k}"] = torch.empty_like(v, device="cpu")

    try:
        dcp.load(state_dict=request_state, checkpoint_id=dcp_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed loading DCP checkpoint from {dcp_path}. "
            "Expected an iter_xxx directory with model/*.distcp files."
        ) from e

    audio_state = {}
    prefix = "student.audio_dit."
    for k, v in request_state.items():
        if k.startswith(prefix):
            audio_state[k[len(prefix):]] = v
    return audio_state


def _load_student_audio_weights(audio_dit, student_ckpt_path):
    if os.path.isdir(student_ckpt_path):
        audio_state = _load_student_audio_weights_from_dcp(audio_dit, student_ckpt_path)
        src_type = "DCP"
    else:
        audio_state = _load_student_audio_weights_from_pt(audio_dit, student_ckpt_path)
        src_type = "PT"

    missing, unexpected = audio_dit.load_state_dict(audio_state, strict=False)
    print(
        f"[*] Loaded student audio weights ({src_type}): {len(audio_state)} tensors, "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )


def load_t2a_models(ckpt_path, device="cuda", student_ckpt_path=None):
    dtype = torch.bfloat16
    print(f"[*] Loading MOVA T2A components from: {ckpt_path}")

    mova_root = _maybe_add_mova_to_syspath()
    if mova_root is None:
        raise ModuleNotFoundError(
            "Could not locate MOVA repo. Set MOVA_ROOT to your MOVA checkout path, "
            "e.g. export MOVA_ROOT=/mnt/16T/home_user/lyx/turbodiffusion/MOVA"
        )

    from mova.diffusion.models import WanAudioModel
    from mova.diffusion.models.dac_vae import DAC
    from mova.diffusion.schedulers.flow_match_pair import FlowMatchPairScheduler

    tokenizer = T5TokenizerFast.from_pretrained(ckpt_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        ckpt_path, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    audio_dit = WanAudioModel.from_pretrained(
        ckpt_path, subfolder="audio_dit", torch_dtype=dtype
    ).to(device)

    if student_ckpt_path:
        print(f"[*] Loading distilled student audio checkpoint: {student_ckpt_path}")
        _load_student_audio_weights(audio_dit, student_ckpt_path)

    audio_vae = DAC.from_pretrained(ckpt_path, subfolder="audio_vae").to(device)
    scheduler = FlowMatchPairScheduler.from_pretrained(ckpt_path, subfolder="scheduler")

    text_encoder.eval()
    audio_dit.eval()
    audio_vae.eval()

    return tokenizer, text_encoder, audio_dit, audio_vae, scheduler


def _get_text_embeds(tokenizer, text_encoder, text_str, device):
    text_inputs = tokenizer(
        [text_str],
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        embeds = text_encoder(
            text_inputs.input_ids.to(device),
            text_inputs.attention_mask.to(device),
        ).last_hidden_state

    seq_lens = text_inputs.attention_mask.gt(0).sum(dim=1)
    embeds = [u[:v] for u, v in zip(embeds, seq_lens)]
    embeds = torch.stack(
        [torch.cat([u, u.new_zeros(512 - u.size(0), u.size(1))]) for u in embeds],
        dim=0,
    )
    return embeds


@torch.no_grad()
def generate_audio(
    prompt,
    models,
    negative_prompt="noise, distorted, low quality, bad audio, muffled, static",
    num_frames=193,
    video_fps=24.0,
    num_inference_steps=50,
    cfg_scale=5.0,
    seed=42,
    device="cuda",
):
    dtype = torch.bfloat16
    tokenizer, text_encoder, audio_dit, audio_vae, scheduler = models

    prompt_embeds = _get_text_embeds(tokenizer, text_encoder, prompt, device)
    neg_embeds = _get_text_embeds(tokenizer, text_encoder, negative_prompt, device)

    latent_t = (int(audio_vae.sample_rate * num_frames / video_fps) - 1) // int(audio_vae.hop_length) + 1
    generator = torch.Generator(device=device).manual_seed(seed)
    audio_latents = torch.randn(
        (1, audio_vae.latent_dim, latent_t),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    scheduler.set_timesteps(num_inference_steps, device=device)
    paired_timesteps = (
        scheduler.get_pairs() if hasattr(scheduler, "get_pairs")
        else torch.stack([scheduler.timesteps, scheduler.timesteps], dim=1)
    )
    actual_steps = int(paired_timesteps.shape[0])

    def _sync_if_cuda():
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(torch.device(device))

    _sync_if_cuda()
    t_start = time.perf_counter()

    with torch.autocast("cuda", dtype=dtype):
        for idx_step in range(paired_timesteps.shape[0]):
            _, audio_t = paired_timesteps[idx_step]
            t_tensor = audio_t.unsqueeze(0).to(device=device, dtype=dtype)

            n_neg = audio_dit(
                x=audio_latents.to(dtype),
                timestep=t_tensor,
                context=neg_embeds.to(dtype),
            ).float()
            n_pos = audio_dit(
                x=audio_latents.to(dtype),
                timestep=t_tensor,
                context=prompt_embeds.to(dtype),
            ).float()
            noise_pred = n_neg + cfg_scale * (n_pos - n_neg)

            next_t = paired_timesteps[idx_step + 1, 1] if idx_step + 1 < paired_timesteps.shape[0] else None
            if hasattr(scheduler, "step_from_to"):
                audio_latents = scheduler.step_from_to(noise_pred, audio_t, next_t, audio_latents)
            else:
                audio_latents = scheduler.step(noise_pred, audio_t, audio_latents, return_dict=False)[0]

    with torch.autocast("cuda", dtype=torch.float32):
        audio_waveform = audio_vae.decode(audio_latents)

    _sync_if_cuda()
    t_total = time.perf_counter() - t_start

    stats = {
        "steps": actual_steps,
        "elapsed_sec": t_total,
        "sec_per_step": (t_total / max(actual_steps, 1)),
    }
    return audio_waveform, audio_vae.sample_rate, stats


def _save_audio_tensor(audio_tensor, save_path, sample_rate):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    audio_data = audio_tensor.squeeze().cpu().to(torch.float32).numpy()
    if len(audio_data.shape) > 1 and audio_data.shape[0] < audio_data.shape[1]:
        audio_data = audio_data.T
    sf.write(save_path, audio_data, sample_rate)


def _read_prompts(prompt_file):
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def main():
    parser = argparse.ArgumentParser(description="MOVA T2A inference with optional distilled student audio weights")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_prefix", type=str, default="sample_")
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--write_prompt_txt", action="store_true")
    parser.add_argument("--student_ckpt_path", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default="noise, distorted, low quality, bad audio, muffled, static")
    parser.add_argument("--num_frames", type=int, default=193)
    parser.add_argument("--video_fps", type=float, default=24.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    is_batch_mode = args.prompt_file is not None
    if is_batch_mode:
        if not os.path.isfile(args.prompt_file):
            raise FileNotFoundError(f"prompt_file not found: {args.prompt_file}")
        if not args.output_dir:
            raise ValueError("output_dir is required when prompt_file is set")
    else:
        if not args.prompt:
            raise ValueError("prompt is required in single inference mode")
        if not args.save_path:
            raise ValueError("save_path is required in single inference mode")

    print("[*] Inference config:")
    print(f"    ckpt_path={args.ckpt_path}")
    print(f"    student_ckpt_path={args.student_ckpt_path}")
    print(f"    device={args.device}")
    print(
        "    num_inference_steps="
        f"{args.num_inference_steps}, cfg_scale={args.cfg_scale}, "
        f"num_frames={args.num_frames}, video_fps={args.video_fps}, seed={args.seed}"
    )
    if is_batch_mode:
        print(f"    prompt_file={args.prompt_file}")
        print(f"    output_dir={args.output_dir}")
        print(f"    seed_base={args.seed_base}")
    else:
        prompt_preview = args.prompt if len(args.prompt) <= 200 else args.prompt[:197] + "..."
        print(f"    save_path={args.save_path}")
        print(f"    prompt={prompt_preview}")

    models = load_t2a_models(
        ckpt_path=args.ckpt_path,
        device=args.device,
        student_ckpt_path=args.student_ckpt_path,
    )

    if is_batch_mode:
        prompts = _read_prompts(args.prompt_file)
        if not prompts:
            raise ValueError(f"No valid prompts in prompt_file: {args.prompt_file}")

        os.makedirs(args.output_dir, exist_ok=True)
        total_elapsed = 0.0
        width = max(2, len(str(len(prompts) - 1)))

        for i, prompt in enumerate(prompts):
            seed = args.seed_base + i
            idx = f"{i:0{width}d}"
            save_path = os.path.join(args.output_dir, f"{args.save_prefix}{idx}.wav")
            txt_path = os.path.join(args.output_dir, f"{args.save_prefix}{idx}.txt")

            print(f"[*] ({i + 1}/{len(prompts)}) seed={seed} -> {save_path}")
            audio, sr, stats = generate_audio(
                prompt=prompt,
                models=models,
                negative_prompt=args.negative_prompt,
                num_frames=args.num_frames,
                video_fps=args.video_fps,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
                seed=seed,
                device=args.device,
            )
            _save_audio_tensor(audio, save_path, sr)
            if args.write_prompt_txt:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(prompt + "\n")

            total_elapsed += stats["elapsed_sec"]
            print(
                "[*] Inference timing: "
                f"steps={stats['steps']}, total={stats['elapsed_sec']:.3f}s, "
                f"per_step={stats['sec_per_step']:.3f}s"
            )

        print(
            "[+] Batch inference done: "
            f"samples={len(prompts)}, total_elapsed={total_elapsed:.3f}s, output_dir={args.output_dir}"
        )
    else:
        audio, sr, stats = generate_audio(
            prompt=args.prompt,
            models=models,
            negative_prompt=args.negative_prompt,
            num_frames=args.num_frames,
            video_fps=args.video_fps,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
            device=args.device,
        )

        _save_audio_tensor(audio, args.save_path, sr)
        print(
            "[*] Inference timing: "
            f"steps={stats['steps']}, total={stats['elapsed_sec']:.3f}s, "
            f"per_step={stats['sec_per_step']:.3f}s"
        )
        print(f"[+] Saved audio to: {args.save_path}")


if __name__ == "__main__":
    main()
