import os
import io
import sys
import math
import tarfile
import time
import torch
import torch.distributed as dist
import argparse
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy
from contextlib import nullcontext

# Add MOVA to sys.path（兼容节点挂载为 gy2 或 gy2_302507476）
_mova_base = "/apdcephfs_gy2/share_302507476/xiaodayang/AudioSummary/MOVA"
if not os.path.exists(_mova_base):
    _mova_base = "/apdcephfs_gy2_302507476/share_302507476/xiaodayang/AudioSummary/MOVA"
sys.path.append(_mova_base)

from imaginaire.utils import distributed
from PIL import Image
from mova.diffusion.pipelines.pipeline_mova import MOVA
from mova.datasets.transforms.custom import crop_and_resize

# Default negative prompt from MOVA inference script
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

tensor_kwargs = {"device": "cuda", "dtype": torch.bfloat16}

def _tensor_is_valid(t):
    """检查 tensor 是否含 NaN/Inf，用于判定 latent 是否合法。"""
    if not isinstance(t, torch.Tensor):
        return True
    return not (torch.isnan(t).any().item() or torch.isinf(t).any().item())

def _load_and_validate_sample_tensors(tar, index, members_by_name, rank, log_prefix=""):
    """从已打开的 tar 中加载某样本的 4 个 tensor 并校验，全部合法返回 True。"""
    for key in ["v_latent", "a_latent", "ref", "embed"]:
        name = f"{index:09d}.{key}.pt"
        if name not in members_by_name:
            return False
        f = tar.extractfile(members_by_name[name])
        if f is None:
            return False
        try:
            t = torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=False)
        except Exception:
            return False
        if not _tensor_is_valid(t):
            if log_prefix:
                print(f"[Rank {rank}] {log_prefix} sample {index} has NaN/Inf in {key}, will re-process.")
            return False
    return True

def validate_shard(shard_path, rank=0):
    """检查 shard tar 内所有样本的 tensor 是否均无 NaN/Inf，合法返回 True。"""
    try:
        with tarfile.open(shard_path, "r") as tar:
            files_in_tar = defaultdict(list)
            members_by_name = {}
            for member in tar.getmembers():
                members_by_name[member.name] = member
                parts = member.name.split(".")
                if len(parts) >= 3 and parts[0].isdigit():
                    idx_str = parts[0]
                    if "v_latent" in member.name:
                        files_in_tar[int(idx_str)].append("v_latent")
                    if "a_latent" in member.name:
                        files_in_tar[int(idx_str)].append("a_latent")
                    if "ref" in member.name:
                        files_in_tar[int(idx_str)].append("ref")
                    if "embed" in member.name:
                        files_in_tar[int(idx_str)].append("embed")
                    if "prompt" in member.name:
                        files_in_tar[int(idx_str)].append("prompt")
            for index, types in files_in_tar.items():
                if not all(k in types for k in ["v_latent", "a_latent", "ref", "embed", "prompt"]):
                    continue
                if not _load_and_validate_sample_tensors(
                    tar, index, members_by_name, rank, log_prefix="Shard"
                ):
                    return False
        return True
    except (tarfile.ReadError, EOFError, OSError, Exception):
        return False

def is_shard_done(shard_path):
    """仅判断文件是否存在且非空；内容合法性由 validate_shard 单独检查。"""
    return os.path.exists(shard_path) and os.path.getsize(shard_path) > 0

def write_to_tar(tar, key, data_bytes):
    ti = tarfile.TarInfo(key)
    ti.size = len(data_bytes)
    tar.addfile(ti, io.BytesIO(data_bytes))

def barrier():
    # torchrun 多进程同步点；不要用死循环占用 GPU 显存
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

@torch.no_grad()
def sample_latents_mova(pipe, prompt, image, negative_prompt, args, device):
    """
    Runs the MOVA inference loop but returns latents and embeddings instead of decoded video/audio.
    Adapted from MOVA.__call__
    """
    height = args.height
    width = args.width
    num_frames = args.num_frames
    video_fps = args.fps
    num_inference_steps = args.num_inference_steps
    sigma_shift = args.sigma_shift
    cfg_scale = args.cfg_scale
    seed = args.seed
    
    # 1. Check inputs
    pipe.check_inputs(height, width, num_frames)
    
    denoising_strength = 1.0
    cfg_mode = "text"
    cfg_merge = False
    audio_num_samples = int(pipe.audio_sample_rate * num_frames / video_fps)
    
    # 4. Prepare timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    scheduler_support_audio = hasattr(pipe.scheduler, "get_pairs")
    if scheduler_support_audio:
        audio_scheduler = pipe.scheduler
        paired_timesteps = pipe.scheduler.get_pairs()
    else:
        audio_scheduler = deepcopy(pipe.scheduler)
        paired_timesteps = torch.stack([pipe.scheduler.timesteps, pipe.scheduler.timesteps], dim=1)
        
    # 5. Prepare latent variables
    num_channels_latents = pipe.video_vae.config.z_dim
    # Image preprocessing
    image = pipe.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32) # 预处理参考图像
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    latents, condition = pipe.prepare_latents(#占位
        image,
        1, # batch_size
        num_channels_latents,
        height,
        width,
        num_frames,
        torch.float32,
        device,
        generator=generator,
    )
    
    # NOTE: MOVA visual DiT expects input channels = video_latents(16) + condition(20) = 36.
    # `condition` returned by `prepare_latents` is the full 20-channel conditioning tensor:
    #   condition = concat([mask_lat_size (scale_factor_temporal channels), latent_condition (16 channels)], dim=1)
    # DO NOT slice it; training needs the full condition tensor.
    ref_latent = condition
    
    audio_latents = pipe.prepare_audio_latents(#占位
        None,
        1,
        pipe.audio_vae.latent_dim,
        audio_num_samples,
        torch.float32,
        device,
        generator=generator,
    )
    
    # Text embeddings
    prompt_embeds = pipe._get_t5_prompt_embeds(prompt, device=device, dtype=pipe.text_encoder.dtype) # 提取文本 Embedding
    negative_prompt_embeds = pipe._get_t5_prompt_embeds(negative_prompt, device=device, dtype=pipe.text_encoder.dtype)
    
    # --------------------------------------------------
    # diffusion steps
    # --------------------------------------------------
    cur_visual_dit = pipe.video_dit
    total_steps = paired_timesteps.shape[0]
    switched = False
    boundary_timestep = pipe.boundary_ratio * pipe.scheduler.config.num_train_timesteps
    remove_video_dit = False # Default from inference script
    
    for idx_step in range(total_steps):
        timestep, audio_timestep = paired_timesteps[idx_step]
        
        # Switch to low-noise DiT logic
        if not switched and timestep.item() < boundary_timestep:
            cur_visual_dit = pipe.video_dit_2
            if remove_video_dit:
                pipe.video_dit = None
                import gc
                gc.collect()
            switched = True
            
        latent_model_input = torch.cat([latents, condition], dim=1)
        timestep_tensor = timestep.unsqueeze(0).to(device=device, dtype=torch.float32)
        audio_timestep_tensor = audio_timestep.unsqueeze(0).to(device=device, dtype=torch.float32)
        
        # Inference single step
        with torch.no_grad():
            noise_pred_posi = pipe.inference_single_step(
                visual_dit=cur_visual_dit,
                visual_latents=latent_model_input,
                audio_latents=audio_latents,
                context=prompt_embeds,
                timestep=timestep_tensor,
                audio_timestep=audio_timestep_tensor,
                video_fps=video_fps,
                cp_mesh=None
            )
            
            # CFG
            if cfg_scale > 1.0:
                noise_pred_nega = pipe.inference_single_step(
                    visual_dit=cur_visual_dit,
                    visual_latents=latent_model_input,
                    audio_latents=audio_latents,
                    context=negative_prompt_embeds,
                    timestep=timestep_tensor,
                    audio_timestep=audio_timestep_tensor,
                    video_fps=video_fps,
                    cp_mesh=None
                )
                visual_noise_pred_nega, audio_noise_pred_nega = noise_pred_nega[0].float(), noise_pred_nega[1].float()
                visual_noise_pred_posi, audio_noise_pred_posi = noise_pred_posi[0].float(), noise_pred_posi[1].float()
                
                visual_noise_pred = visual_noise_pred_nega + cfg_scale * (visual_noise_pred_posi - visual_noise_pred_nega)
                audio_noise_pred = audio_noise_pred_nega + cfg_scale * (audio_noise_pred_posi - audio_noise_pred_nega)
            else:
                visual_noise_pred = noise_pred_posi[0].float()
                audio_noise_pred = noise_pred_posi[1].float()
                
            # Step
            if scheduler_support_audio:
                next_timestep = paired_timesteps[idx_step + 1, 0] if idx_step + 1 < total_steps else None
                next_audio_timestep = paired_timesteps[idx_step + 1, 1] if idx_step + 1 < total_steps else None
                latents = pipe.scheduler.step_from_to(
                    visual_noise_pred, timestep, next_timestep, latents
                )
                audio_latents = audio_scheduler.step_from_to(
                    audio_noise_pred, audio_timestep, next_audio_timestep, audio_latents
                )
            else:
                latents = pipe.scheduler.step(visual_noise_pred, timestep, latents, return_dict=False)[0]
                audio_latents = audio_scheduler.step(audio_noise_pred, audio_timestep, audio_latents, return_dict=False)[0]

    return latents, audio_latents, ref_latent, prompt_embeds

def main(args):
    distributed.init() # 初始化分布式环境 (依赖 torchrun 设置的环境变量)
    rank = distributed.get_rank() # 获取当前进程的 rank
    world_size = distributed.get_world_size() # 获取总进程数
    
    items = []
    
    # Logic for input_dir
    if args.input_dir:
        if not os.path.exists(args.input_dir):
            raise FileNotFoundError(f"[Rank {rank}] Input directory does not exist: {args.input_dir}")
        
        # Walk through subdirectories
        # Assuming structure: input_dir/sub_folder/prompt.txt & image.(png|jpg|jpeg|webp)
        subdirs = sorted(
            [
                os.path.join(args.input_dir, d)
                for d in os.listdir(args.input_dir)
                if os.path.isdir(os.path.join(args.input_dir, d))
            ]
        )

        # 支持多种图片后缀：.png, .jpg, .jpeg, .webp
        IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
        for subdir in subdirs:
            prompt_path = os.path.join(subdir, "prompt.txt")
            image_path = None
            for ext in IMAGE_EXTENSIONS:
                candidate = os.path.join(subdir, f"image{ext}")
                if os.path.exists(candidate):
                    image_path = candidate
                    break

            if os.path.exists(prompt_path) and image_path is not None:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    prompt = f.read().strip()
                if prompt:
                    items.append((image_path, prompt))
            else:
                # print(f"[Rank {rank}] Skipping {subdir}: missing prompt.txt or image.(png/jpg/jpeg/webp)")
                pass
                
    elif args.prompt_file:
        # Existing logic
        if not os.path.exists(args.prompt_file):
             print(f"[Rank {rank}] Prompt file {args.prompt_file} does not exist.")
             return

        with open(args.prompt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "||" in line:
                    ref_p, p = line.split("||", 1)
                    items.append((ref_p.strip(), p.strip()))
                else:
                    items.append((args.ref_path, line))
    else:
        if rank == 0:
            print("Either --input_dir or --prompt_file must be provided.")
        return
                
    if not items:
        if rank == 0:
            print(f"No items found to process.")
        return

    # Repeat items
    if args.repeat > 1:
        items = items * args.repeat
        
    total = len(items)
    num_shards = math.ceil(total / args.samples_per_shard)
    my_shards = [i for i in range(num_shards) if i % world_size == rank] # 当前进程负责的分片
    
    print(f"[Rank {rank}] Read {total} items, total {num_shards} shards.")
    print(f"[Rank {rank}] will build {len(my_shards)} shards.")
    
    # 如果该 rank 没有任何 shard 要做，就不要白白加载 MOVA（会占用大量显存）
    if not my_shards:
        print(f"[Rank {rank}] No shards assigned. Skip model loading.")
        barrier()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        return

    # Load MOVA Model
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)
    
    pipe = MOVA.from_pretrained(args.ckpt_path, torch_dtype=torch.bfloat16) # 加载 MOVA 模型
    # 和 MOVA 官方推理脚本一致的 offload 策略（默认 none）
    if args.offload == "none":
        pipe.to(device)
    elif args.offload == "cpu":
        pipe.enable_model_cpu_offload(local_rank)
    elif args.offload == "group":
        pipe.enable_group_offload(
            onload_device=torch.device("cuda", local_rank),
            offload_device=torch.device("cpu"),
            offload_type="leaf_level",
            use_stream=True,
            low_cpu_mem_usage=True,
            offload_to_disk_path=args.offload_to_disk_path,
        )
    else:
        raise ValueError(f"Unknown offload strategy: {args.offload}")
    
    num_my_shards = len(my_shards)
    shard_iter = tqdm(
        my_shards,
        desc="Shards",
        total=num_my_shards,
        unit="shard",
        disable=(rank != 0),
        position=0,
    )
    for shard_id in shard_iter:
        shard_iter.set_postfix_str(f"shard_{shard_id:06d}")
        shard_path = os.path.join(args.output_dir, f"shard_{shard_id:06d}.tar")
        
        if is_shard_done(shard_path):
            if validate_shard(shard_path, rank):
                print(f"[Rank {rank}] Shard {shard_id} already completed and valid. Skip.")
                continue
            print(f"[Rank {rank}] Shard {shard_id} exists but contains invalid (e.g. NaN/Inf) tensors. Remove and rebuild.")
            try:
                os.remove(shard_path)
            except OSError:
                pass
            # 若有残留 .tmp 也删掉，从头建
            tmp_path_check = shard_path + ".tmp"
            if os.path.exists(tmp_path_check):
                try:
                    os.remove(tmp_path_check)
                except OSError:
                    pass

        start = shard_id * args.samples_per_shard
        end = min(total, start + args.samples_per_shard)
        
        print(f"[Rank {rank}] Building shard {shard_id}, items {start}..{end-1}")
        
        tmp_path = shard_path + ".tmp"
        
        processed_indices = set()
        if os.path.exists(tmp_path):
            try:
                tmp_has_invalid = False
                with tarfile.open(tmp_path, "r") as existing_tar:
                    files_in_tar = defaultdict(list)
                    members_by_name = {}
                    for member in existing_tar.getmembers():
                        members_by_name[member.name] = member
                        parts = member.name.split(".")
                        if len(parts) >= 3 and parts[0].isdigit():
                            index_str = parts[0]
                            if "v_latent" in member.name:
                                files_in_tar[int(index_str)].append("v_latent")
                            if "a_latent" in member.name:
                                files_in_tar[int(index_str)].append("a_latent")
                            if "ref" in member.name:
                                files_in_tar[int(index_str)].append("ref")
                            if "embed" in member.name:
                                files_in_tar[int(index_str)].append("embed")
                            if "prompt" in member.name:
                                files_in_tar[int(index_str)].append("prompt")

                    for index, types in files_in_tar.items():
                        if not all(k in types for k in ["v_latent", "a_latent", "ref", "embed", "prompt"]):
                            continue
                        if _load_and_validate_sample_tensors(
                            existing_tar, index, members_by_name, rank, log_prefix="Tmp shard"
                        ):
                            processed_indices.add(index)
                        else:
                            tmp_has_invalid = True
                            break
                if tmp_has_invalid:
                    processed_indices = set()
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                    print(f"[Rank {rank}] Tmp shard has invalid sample(s). Removed tmp, will rebuild shard from scratch.")
                elif processed_indices:
                    print(f"[Rank {rank}] Found {len(processed_indices)} completed and valid items in tmp file.")
            except (tarfile.ReadError, EOFError, OSError) as e:
                raise RuntimeError(
                    f"[Rank {rank}] Failed to read existing tmp shard file (may be corrupted): {tmp_path}. "
                    f"Please delete it and rerun. Original error: {e}"
                ) from e
        
        # Open tar in append mode
        with tarfile.open(tmp_path, "a") as tar:
            for idx in tqdm(
                range(start, end),
                desc=f"Shard {shard_id}",
                total=end - start,
                initial=len(processed_indices),
                unit="sample",
                disable=(rank != 0),
                position=1,
                leave=False,
            ):
                if idx in processed_indices:
                    continue
                    
                ref_path, prompt = items[idx]

                img = Image.open(ref_path).convert("RGB")
                img = crop_and_resize(img, height=args.height, width=args.width)

                args.seed = idx  # Use index as seed

                v_latent, a_latent, ref_latent, text_embed = sample_latents_mova(
                    pipe, prompt, img, args.negative_prompt, args, device
                ) # 采样获取 Latent

                key_prefix = f"{idx:09d}"

                buf = io.BytesIO()
                torch.save(v_latent.cpu(), buf)
                write_to_tar(tar, f"{key_prefix}.v_latent.pt", buf.getvalue())

                buf = io.BytesIO()
                torch.save(a_latent.cpu(), buf)
                write_to_tar(tar, f"{key_prefix}.a_latent.pt", buf.getvalue())

                buf = io.BytesIO()
                torch.save(ref_latent.cpu(), buf)
                write_to_tar(tar, f"{key_prefix}.ref.pt", buf.getvalue())

                buf = io.BytesIO()
                torch.save(text_embed.cpu(), buf)
                write_to_tar(tar, f"{key_prefix}.embed.pt", buf.getvalue())

                write_to_tar(tar, f"{key_prefix}.prompt.txt", prompt.encode("utf-8"))
                    
        os.rename(tmp_path, shard_path)
        print(f"[Rank {rank}] Finished shard {shard_id}")
        
    print(f"[Rank {rank}] All done.")
    barrier()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Directory containing subfolders with prompt.txt and image.png")
    parser.add_argument("--prompt_file", type=str, help="Legacy: Text file with prompts")
    parser.add_argument("--ref_path", type=str, help="Default reference image path if not specified in prompt file")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    
    parser.add_argument("--samples_per_shard", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=1)
    
    # MOVA params
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_frames", type=int, default=81) 
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--negative_prompt", type=str, default=NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, default=42) # Base seed

    parser.add_argument("--offload", type=str, default="none", choices=("none", "cpu", "group"))
    parser.add_argument("--offload_to_disk_path", type=str, default=None)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
