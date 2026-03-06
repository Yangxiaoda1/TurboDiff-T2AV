import argparse
import os

import torch
import torch.distributed as dist
from PIL import Image
from torch.distributed.device_mesh import DeviceMesh
from yunchang.kernels import AttnType

from mova.datasets.transforms.custom import crop_and_resize
from mova.diffusion.pipelines.pipeline_mova import MOVA
from mova.utils.data import save_video_with_audio

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指"
)

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/MOVA-720p")

    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default=NEGATIVE_PROMPT)
    parser.add_argument("--ref_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./data/samples/output.mp4")
    parser.add_argument("--num_frames", type=int, default=193)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--sigma_shift", type=float, default=5.0)

    parser.add_argument("--cp_size", type=int, default=1)
    parser.add_argument(
        "--attn_type", type=str, default="fa", choices=tuple(t.value for t in AttnType)
    )
    parser.add_argument("--offload", type=str, default="group", choices=("none", "cpu", "group"))
    parser.add_argument("--offload_to_disk_path", type=str, default=None)
    parser.add_argument("--remove_video_dit", action="store_true")

    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.ref_path):
        raise FileNotFoundError(args.ref_path)

    # --------------------------------------------------
    # Init distributed
    # --------------------------------------------------
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    torch.cuda.set_device(local_rank)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    cp_size = args.cp_size
    dp_size = world_size // cp_size
    mesh = DeviceMesh(
        "cuda",
        torch.arange(dist.get_world_size()).view(dp_size, cp_size),
        mesh_dim_names=("dp", "cp"),
    )

    torch.manual_seed(args.seed)
    torch_dtype = torch.bfloat16


    pipe = MOVA.from_pretrained(args.ckpt_path, torch_dtype=torch_dtype)
    # --------------------------------------------------
    # Offload strategy
    # --------------------------------------------------
    if args.offload == "none":
        pipe.to(torch.device("cuda", local_rank))
    elif args.offload == "cpu":
        pipe.enable_model_cpu_offload(local_rank)
    elif args.offload == "group":
        onload_device = torch.device("cuda", local_rank)
        offload_device = torch.device("cpu")
        pipe.enable_group_offload(
            onload_device=onload_device,
            offload_device=offload_device,
            offload_type="leaf_level",
            use_stream=True,
            low_cpu_mem_usage=True,
            offload_to_disk_path=args.offload_to_disk_path,
        )
    else:
        raise ValueError(f"Unknown offload strategy: {args.offload}")

    # --------------------------------------------------
    # Context Parallel
    # --------------------------------------------------
    if args.cp_size > 1:
        from yunchang import set_seq_parallel_pg

        MAX_ULYSSES_DEGREE = 4
        ULYSSES_DEGREE = min(args.cp_size, MAX_ULYSSES_DEGREE)
        assert args.cp_size % ULYSSES_DEGREE == 0, "args.cp_size must be divisible by ULYSSES_DEGREE"

        set_seq_parallel_pg(
            sp_ulysses_degree=ULYSSES_DEGREE,
            sp_ring_degree=args.cp_size // ULYSSES_DEGREE,
            rank=rank,
            world_size=world_size,
            use_ulysses_low=True,
        )

        replaced = pipe.replace_attention(attn_type=AttnType.from_string(args.attn_type))
        if is_main_process():
            print(f"[CP] Enabled, replaced {replaced} attention modules")

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    img = Image.open(args.ref_path).convert("RGB")
    ref_img = crop_and_resize(img, height=args.height, width=args.width)

    if is_main_process():
        print("Starting inference...")

    video, audio = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_frames=args.num_frames,
        image=ref_img,
        height=args.height,
        width=args.width,
        video_fps=args.fps,
        num_inference_steps=args.num_inference_steps,
        sigma_shift=args.sigma_shift,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        cp_mesh=mesh["cp"],
        remove_video_dit=args.remove_video_dit,
    )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------
    # rank-0 only
    if is_main_process():
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        audio_save = audio[0].cpu().squeeze()

        save_video_with_audio(
            video[0],
            audio_save,
            args.output_path,
            fps=args.fps,
            sample_rate=pipe.audio_sample_rate,
            quality=9,
        )

        print(f"Saved to {args.output_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
