""" 
Copyright (c) 2025 by TurboDiffusion team.

Licensed under the Apache License, Version 2.0 (the "License");

Citation (please cite if you use this code):

@article{zhang2025turbodiffusion,
  title={TurboDiffusion: Accelerating Video Diffusion Models by 100-200 Times},
  author={Zhang, Jintao and Zheng, Kaiwen and Jiang, Kai and Wang, Haoxu and Stoica, Ion and Gonzalez, Joseph E and Chen, Jianfei and Zhu, Jun},
  journal={arXiv preprint arXiv:2512.16093},
  year={2025}
}
"""

import argparse # 用于解析命令行参数的库

import torch # PyTorch 深度学习框架
from rcm.utils.model_utils import load_state_dict # 从 rcm 模块导入加载权重的工具函数
from rcm.networks.wan2pt1 import ( # 导入 Wan2.1 模型的各个组件
    WanModel as WanModel2pt1,
    WanLayerNorm as WanLayerNorm2pt1,
    WanRMSNorm as WanRMSNorm2pt1,
    WanSelfAttention as WanSelfAttention2pt1
)
from rcm.networks.wan2pt2 import ( # 导入 Wan2.2 模型的各个组件
    WanModel as WanModel2pt2,
    WanLayerNorm as WanLayerNorm2pt2,
    WanRMSNorm as WanRMSNorm2pt2,
    WanSelfAttention as WanSelfAttention2pt2
)

from ops import FastLayerNorm, FastRMSNorm, Int8Linear # 从 ops 导入加速算子：快速归一化和 8 位量化线性层
from SLA import ( # 从 SLA 导入稀疏线性注意力算子
    SparseLinearAttention as SLA,
    SageSparseLinearAttention as SageSLA
)


def replace_attention( # 函数：替换模型中的注意力机制
    model: torch.nn.Module, # 输入模型
    attention_type: str, # 注意力类型：'sla' 或 'sagesla'
    sla_topk: float, # SLA 算法中的 top-k 比例
) -> torch.nn.Module:
    assert attention_type in ["sla", "sagesla"], "无效的注意力类型。" # 断言检查类型是否合法
    
    for module in model.modules(): # 遍历模型中的所有模块
        # 如果是 Wan2.1 或 Wan2.2 的自注意力模块
        if type(module) is WanSelfAttention2pt1 or type(module) is WanSelfAttention2pt2:
            if attention_type == "sla": # 如果指定使用 SLA
                # 替换本地注意力算子为 SLA
                module.attn_op.local_attn = SLA(head_dim=module.dim // module.num_heads, topk=sla_topk, BLKQ=128, BLKK=64)
            elif attention_type == "sagesla": # 如果指定使用 SageSLA
                # 替换本地注意力算子为 SageSLA
                module.attn_op.local_attn = SageSLA(head_dim=module.dim // module.num_heads, topk=sla_topk)
    return model # 返回修改后的模型


def replace_linear_norm( # 函数：替换模型中的线性层和归一化层
    model: torch.nn.Module, # 输入模型
    replace_linear: bool = False, # 是否替换线性层（量化）
    replace_norm: bool = False, # 是否替换归一化层（快速算子）
    quantize: bool = True, # 是否在转换时进行量化
    skip_layer: str = "proj_l" # 需要跳过不处理的层名称关键字
) -> torch.nn.Module:
    replacements = {} # 存储需要替换的映射关系 {名称: 新模块}
    for name, module in model.blocks.named_modules(): # 遍历模型 blocks 中的子模块
        if isinstance(module, torch.nn.Linear) and replace_linear: # 如果是线性层且需要替换
            if skip_layer not in name: # 且不在跳过列表中
                # 将普通 Linear 转换为 Int8Linear
                replacements[name] = Int8Linear.from_linear(module, quantize)
        
        # 替换 RMSNorm 为 FastRMSNorm
        if (isinstance(module, WanRMSNorm2pt1) or isinstance(module, WanRMSNorm2pt2)) and replace_norm:
            replacements[name] = FastRMSNorm.from_rmsnorm(module)
        
        # 替换 LayerNorm 为 FastLayerNorm
        if (isinstance(module, WanLayerNorm2pt1) or isinstance(module, WanLayerNorm2pt2)) and replace_norm:
            replacements[name] = FastLayerNorm.from_layernorm(module)

    for name, new_module in replacements.items(): # 执行替换操作
        parent_module = model.blocks # 从 blocks 开始查找
        name_parts = name.split(".") # 分解模块路径
        for part in name_parts[:-1]: # 逐层深入到父模块
            parent_module = getattr(parent_module, part)
        setattr(parent_module, name_parts[-1], new_module) # 将旧模块替换为新模块
    return model # 返回修改后的模型


tensor_kwargs = {"device": "cuda", "dtype": torch.bfloat16} # 定义默认的张量配置：GPU 加载，BFloat16 精度

def select_model(model_name: str) -> torch.nn.Module: # 函数：根据名称选择并实例化模型架构
    if model_name == "Wan2.1-1.3B": # Wan2.1 1.3B 参数配置
        return WanModel2pt1(
            dim=1536,
            eps=1e-06,
            ffn_dim=8960,
            freq_dim=256,
            in_dim=16,
            model_type="t2v",
            num_heads=12,
            num_layers=30,
            out_dim=16,
            text_len=512,
        )
    elif model_name == "Wan2.1-14B": # Wan2.1 14B 参数配置
        return WanModel2pt1(
            dim=5120,
            eps=1e-06,
            ffn_dim=13824,
            freq_dim=256,
            in_dim=16,
            model_type="t2v",
            num_heads=40,
            num_layers=40,
            out_dim=16,
            text_len=512,
        )
    elif model_name == "Wan2.2-A14B": # Wan2.2 14B 参数配置 (i2v)
        return WanModel2pt2(
            dim=5120,
            eps=1e-06,
            ffn_dim=13824,
            freq_dim=256,
            in_dim=36,
            model_type="i2v",
            num_heads=40,
            num_layers=40,
            out_dim=16,
            text_len=512,
        )
    else:
        raise ValueError(f"未知模型名称: {model_name}")


def create_model(dit_path: str, args: argparse.Namespace) -> torch.nn.Module: # 函数：创建并初始化推理模型
    with torch.device("meta"): # 使用 meta 设备创建模型架子，不占用显存
        net = select_model(args.model)

    state_dict = load_state_dict(dit_path) # 从磁盘加载权重文件
    if args.attention_type in ['sla', 'sagesla']: # 如果指定了加速注意力
        net = replace_attention(net, attention_type=args.attention_type, sla_topk=args.sla_topk) # 替换注意力模块
    # 替换线性层（量化）和归一化层
    replace_linear_norm(net, replace_linear=args.quant_linear, replace_norm=not args.default_norm, quantize=False)
    net.load_state_dict(state_dict, assign=True) # 将权重加载到修改后的模型中，assign=True 直接分配 tensor
    net = net.to(tensor_kwargs["device"]).eval() # 移动到 GPU 并设置为评估模式
    del state_dict # 释放权重字典占用的内存
    return net # 返回最终可用的模型


def parse_arguments() -> argparse.Namespace: # 函数：解析命令行参数（用于独立运行此脚本进行模型转换）
    parser = argparse.ArgumentParser(description="TurboDiffusion 替换注意力模块并量化模型")
    parser.add_argument("--model", choices=["Wan2.1-1.3B", "Wan2.1-14B", "Wan2.2-A14B"], default="Wan2.1-1.3B", help="使用的模型架构")
    parser.add_argument("--input_path", type=str, default="", help="rCM-SLA 微调后的 Wan 模型 checkpoint 路径")
    parser.add_argument("--output_path", type=str, default="", help="保存修改后模型的路径")
    parser.add_argument("--attention_type", choices=["sla", "sagesla", "original"], default="original", help="注意力机制类型")
    parser.add_argument("--sla_topk", type=float, default=0.2, help="SLA/SageSLA 的 top-k 比例")
    parser.add_argument("--quant_linear", action="store_true", help="是否将线性层替换为量化版本")
    parser.add_argument("--default_norm", action="store_true", help="是否使用默认的归一化层（不替换为快速算子）")
    return parser.parse_args()


if __name__ == "__main__": # 如果脚本作为主程序运行
    args = parse_arguments() # 获取参数
    
    with torch.device("meta"): # 创建模型骨架
        net = select_model(args.model)

    state_dict = load_state_dict(args.input_path)["state_dict"] # 加载原始微调权重

    # 移除权重 key 中的 "net." 前缀以保持兼容性
    prefix_to_load = "net."
    state_dict_dit_compatible = dict()
    for k, v in state_dict.items():
        new_key = k[len(prefix_to_load) :] if k.startswith(prefix_to_load) else k
        # 如果是 patch_embedding 层，可能需要 reshape 形状以匹配
        if k.endswith("patch_embedding.weight"):
            v = v.reshape(net.patch_embedding.weight.shape)
        if k.endswith("patch_embedding.bias"):
            v = v.reshape(net.patch_embedding.bias.shape)
        state_dict_dit_compatible[new_key] = v

    if args.attention_type in ['sla', 'sagesla']: # 替换注意力机制
        net = replace_attention(net, attention_type=args.attention_type, sla_topk=args.sla_topk)
    # 加载兼容处理后的权重
    net.load_state_dict(state_dict_dit_compatible, strict=False, assign=True)
    net = net.to(tensor_kwargs["device"]).eval() # 准备模型环境
    del state_dict, state_dict_dit_compatible # 释放内存

    # 替换线性层（量化）和归一化算子
    net = replace_linear_norm(net, replace_linear=args.quant_linear, replace_norm=not args.default_norm)
    torch.save(net.state_dict(), args.output_path) # 保存最终转换好的模型权重
