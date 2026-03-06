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

"""
Wan2.1 T2V (Text-to-Video) TurboDiffusion 推理脚本。
该脚本支持使用蒸馏后的 DiT 模型进行快速视频生成。
"""

import argparse
import math

import torch
from einops import rearrange, repeat
from tqdm import tqdm

from imaginaire.utils.io import save_image_or_video
from imaginaire.utils import log

from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface

from modify_model import tensor_kwargs, create_model

# 抑制 torch.compile 可能产生的错误
torch._dynamo.config.suppress_errors = True


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Wan2.1 T2V 的 TurboDiffusion 推理脚本")
    parser.add_argument("--dit_path", type=str, required=True, help="蒸馏模型的 DiT 模型权重自定义路径")
    parser.add_argument("--model", choices=["Wan2.1-1.3B", "Wan2.1-14B"], default="Wan2.1-1.3B", help="使用的模型版本")
    parser.add_argument("--num_samples", type=int, default=1, help="生成的样本数量")
    parser.add_argument("--num_steps", type=int, choices=[1, 2, 3, 4], default=4, help="时间步蒸馏推理的步骤数 (1~4)")
    parser.add_argument("--sigma_max", type=float, default=80, help="rCM 的初始 sigma 值")
    parser.add_argument("--vae_path", type=str, default="checkpoints/Wan2.1_VAE.pth", help="Wan2.1 VAE 的路径")
    parser.add_argument("--text_encoder_path", type=str, default="checkpoints/models_t5_umt5-xxl-enc-bf16.pth", help="umT5 文本编码器的路径")
    parser.add_argument("--num_frames", type=int, default=81, help="生成的帧数")
    parser.add_argument("--prompt", type=str, default=None, help="视频生成的文本提示词 (除非使用 --serve 模式，否则必填)")
    parser.add_argument("--resolution", default="480p", type=str, help="生成输出的分辨率")
    parser.add_argument("--aspect_ratio", default="16:9", type=str, help="生成输出的宽高比 (宽:高)")
    parser.add_argument("--seed", type=int, default=0, help="用于可重复性的随机种子")
    parser.add_argument("--save_path", type=str, default="output/generated_video.mp4", help="保存生成视频的路径 (包含文件扩展名)")
    parser.add_argument("--attention_type", choices=["sla", "sagesla", "original"], default="sagesla", help="使用的注意力机制类型")
    parser.add_argument("--sla_topk", type=float, default=0.1, help="SLA/SageSLA 注意力的 top-k 比例")
    parser.add_argument("--quant_linear", action="store_true", help="是否将线性层替换为量化版本")
    parser.add_argument("--default_norm", action="store_true", help="是否将 LayerNorm/RMSNorm 层替换为更快的版本")
    parser.add_argument("--serve", action="store_true", help="启动交互式 TUI 服务器模式 (保持模型加载状态)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # 处理服务器模式
    if args.serve:
        # 为 TUI 服务器设置模式为 t2v
        args.mode = "t2v"
        from serve.tui import main as serve_main
        serve_main(args)
        exit(0)

    # 验证在单次生成模式下是否提供了提示词
    if args.prompt is None:
        log.error("需要提供 --prompt (除非使用 --serve 模式)")
        exit(1)

    # 计算提示词的文本嵌入
    log.info(f"正在为提示词计算嵌入: {args.prompt}")
    with torch.no_grad():
        text_emb = get_umt5_embedding(checkpoint_path=args.text_encoder_path, prompts=args.prompt).to(**tensor_kwargs)
    # 释放 umT5 占用的内存
    clear_umt5_memory()

    # 加载 DiT 模型
    log.info(f"正在从 {args.dit_path} 加载 DiT 模型")
    net = create_model(dit_path=args.dit_path, args=args).cpu()
    torch.cuda.empty_cache()
    log.success("成功加载 DiT 模型。")
    
    # 初始化 VAE 接口
    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)

    # 获取对应分辨率和宽高比的视频尺寸
    w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]

    log.info(f"正在使用提示词生成: {args.prompt}")
    # 准备条件信息
    condition = {"crossattn_emb": repeat(text_emb.to(**tensor_kwargs), "b l d -> (k b) l d", k=args.num_samples)}

    to_show = []

    # 定义潜在空间的形状
    state_shape = [
        tokenizer.latent_ch,
        tokenizer.get_latent_num_frames(args.num_frames),
        h // tokenizer.spatial_compression_factor,
        w // tokenizer.spatial_compression_factor,
    ]

    # 初始化生成器并设置种子
    generator = torch.Generator(device=tensor_kwargs["device"])
    generator.manual_seed(args.seed)

    # 生成初始噪声
    init_noise = torch.randn(
        args.num_samples,
        *state_shape,
        dtype=torch.float32,
        device=tensor_kwargs["device"],
        generator=generator,
    )

    # 设置中间时间步，为了更好的视觉质量进行了调整
    # mid_t = [1.3, 1.0, 0.6][: args.num_steps - 1]
    mid_t = [1.5, 1.4, 1.0][: args.num_steps - 1]

    # 构建时间步序列
    t_steps = torch.tensor(
        [math.atan(args.sigma_max), *mid_t, 0],
        dtype=torch.float64,
        device=init_noise.device,
    )

    # 将 TrigFlow 时间步转换为 RectifiedFlow 时间步
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

    # 采样步骤开始
    x = init_noise.to(torch.float64) * t_steps[0]
    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
    total_steps = t_steps.shape[0] - 1
    net.cuda() # 将模型移至 GPU 进行推理
    for i, (t_cur, t_next) in enumerate(tqdm(list(zip(t_steps[:-1], t_steps[1:])), desc="正在采样", total=total_steps)):
        with torch.no_grad():
            # 预测视频速度向量 v
            v_pred = net(x_B_C_T_H_W=x.to(**tensor_kwargs), timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs), **condition).to(
                torch.float64
            )
            # 更新潜在变量 x，并加入随机性（针对蒸馏推理的特定步骤）
            x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                *x.shape,
                dtype=torch.float32,
                device=tensor_kwargs["device"],
                generator=generator,
            )
    samples = x.float()
    net.cpu() # 推理完成后将模型移回 CPU
    torch.cuda.empty_cache()

    # 使用 VAE 解码潜在样本回视频空间
    with torch.no_grad():
        video = tokenizer.decode(samples)

    to_show.append(video.float().cpu())

    # 将视频归一化到 [0, 1] 范围
    to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0

    # 保存生成的视频
    save_image_or_video(rearrange(to_show, "n b c t h w -> c t (n h) (b w)"), args.save_path, fps=16)
