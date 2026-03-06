import argparse
import os
import sys
import torch
import torch.distributed as dist
from PIL import Image

# Add MOVA to sys.path
sys.path.append("/apdcephfs_gy2/share_302507476/xiaodayang/AudioSummary/MOVA")

from mova.diffusion.pipelines.pipeline_mova import MOVA
from mova.datasets.transforms.custom import crop_and_resize
from mova.utils.data import save_video_with_audio

def main(args):
    # Initialize dummy distributed environment for MOVA pipeline compatibility
    # MOVA pipeline calls dist.get_rank(), so we must init process group.
    # Since we run independent processes per GPU, each is a world of size 1.
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        # Use a unique port based on device_id to avoid conflicts between parallel runs
        os.environ["MASTER_PORT"] = str(29500 + args.device_id + 100) 
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        try:
            dist.init_process_group(backend="nccl", init_method="env://")
        except Exception as e:
            print(f"[GPU {args.device_id}] Warning: Failed to init dist group: {e}")

    device = torch.device(f"cuda:{args.device_id}")
    torch.cuda.set_device(device)

    print(f"[GPU {args.device_id}] Loading MOVA from {args.ckpt_path}...")
    pipe = MOVA.from_pretrained(args.ckpt_path, torch_dtype=torch.bfloat16)
    
    if args.offload:
        pipe.enable_model_cpu_offload(args.device_id)
    else:
        pipe.to(device)

    # Load distilled student weights if provided
    if args.student_ckpt_path:
        print(f"[GPU {args.device_id}] Loading student weights from {args.student_ckpt_path}...")
        student_state_dict = torch.load(args.student_ckpt_path, map_location="cpu")
        
        # Manually distribute weights to sub-modules because Pipeline doesn't support load_state_dict
        video_dit_state = {}
        video_dit_2_state = {}
        audio_dit_state = {}
        bridge_state = {}
        
        for k, v in student_state_dict.items():
            if k.startswith("video_dit."):
                video_dit_state[k.replace("video_dit.", "")] = v
            elif k.startswith("video_dit_2."):
                video_dit_2_state[k.replace("video_dit_2.", "")] = v
            elif k.startswith("audio_dit."):
                audio_dit_state[k.replace("audio_dit.", "")] = v
            elif k.startswith("bridge."):
                bridge_state[k.replace("bridge.", "")] = v
            elif k.startswith("dual_tower_bridge."): 
                bridge_state[k.replace("dual_tower_bridge.", "")] = v
                
        # Load into pipe sub-modules
        if video_dit_state:
            print(f"[GPU {args.device_id}] Loading {len(video_dit_state)} keys into video_dit...")
            m, u = pipe.video_dit.load_state_dict(video_dit_state, strict=False)
            print(f"[GPU {args.device_id}] Video DiT: Missing {len(m)}, Unexpected {len(u)}")
            
        if video_dit_2_state:
            print(f"[GPU {args.device_id}] Loading {len(video_dit_2_state)} keys into video_dit_2...")
            m, u = pipe.video_dit_2.load_state_dict(video_dit_2_state, strict=False)
            print(f"[GPU {args.device_id}] Video DiT 2: Missing {len(m)}, Unexpected {len(u)}")
            
        if audio_dit_state:
            print(f"[GPU {args.device_id}] Loading {len(audio_dit_state)} keys into audio_dit...")
            m, u = pipe.audio_dit.load_state_dict(audio_dit_state, strict=False)
            print(f"[GPU {args.device_id}] Audio DiT: Missing {len(m)}, Unexpected {len(u)}")

        if bridge_state:
            print(f"[GPU {args.device_id}] Loading {len(bridge_state)} keys into dual_tower_bridge...")
            m, u = pipe.dual_tower_bridge.load_state_dict(bridge_state, strict=False)
            print(f"[GPU {args.device_id}] Bridge: Missing {len(m)}, Unexpected {len(u)}")

    prompt = args.prompt
    if not os.path.exists(args.image_path):
        print(f"Image not found: {args.image_path}")
        return
        
    image = Image.open(args.image_path).convert("RGB")
    image = crop_and_resize(image, height=args.height, width=args.width)

    print(f"[GPU {args.device_id}] Generating...")
    
    # Inference
    video_frames, audio_waveform = pipe(
        prompt=prompt,
        image=image,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        video_fps=args.fps,
        num_inference_steps=args.steps,
        sigma_shift=args.sigma_shift,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    # Unwrap batch: pipeline may return list of videos / list of audios
    frames_out = video_frames[0] if isinstance(video_frames, list) and len(video_frames) > 0 and isinstance(video_frames[0], list) else video_frames
    audio_out = audio_waveform[0].cpu().squeeze() if isinstance(audio_waveform, (list, tuple)) else audio_waveform.cpu().squeeze()
    save_video_with_audio(frames_out, audio_out, args.save_path, args.fps, sample_rate=pipe.audio_sample_rate, quality=9)
    print(f"[GPU {args.device_id}] Saved to {args.save_path}")
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--student_ckpt_path", type=str, default=None)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--fps", type=float, default=24.0)
    
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative_prompt", type=str, default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")

    args = parser.parse_args()
    main(args)
