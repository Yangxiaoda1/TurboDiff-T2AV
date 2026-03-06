import torch
import os
import glob
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import gc

# ================= ⚙️ 配置区域 =================

# 1. 输入模型文件夹 (MOVA 原版分卷路径)
# 第一次运行：填MOVA-High Dit 路径 (例如 D:\xunleidownload\mova-video-dit)
# 第二次运行：填MOVA-Low  Dit 路径
# PATH_INPUT_FOLDER = r"D:\xunleidownload\mova-video-dit"
PATH_INPUT_FOLDER = r"D:\Models\dit-2"

# 2. 差值文件路径 (Delta)
# 第一次运行：填 High_Delta.safetensors
# 第二次运行：填 Low_Delta.safetensors
# PATH_DELTA = r"D:\Models\Wan2.2_14B_Turbo_Delta_High.safetensors"
PATH_DELTA = r"D:\Models\Wan2.2_14B_Turbo_Delta_Low.safetensors"

# 3. 输出保存文件夹 (脚本会自动创建)
# 第一次运行建议：D:\Models\MOVA_Turbo_High
# 第二次运行建议：D:\Models\MOVA_Turbo_Low
# PATH_OUTPUT_FOLDER = r"D:\Models\MOVA_Turbo_High"
PATH_OUTPUT_FOLDER = r"D:\Models\MOVA_Turbo_Low"

# 4. 混合强度 (Alpha)
# 1.0 = 完全 Turbo 化
ALPHA = 1.0 

# ===============================================

def clean_key(key):
    """清洗 Key 前缀，确保能匹配上"""
    prefixes = ["model.diffusion_model.", "diffusion_model.", "module.", "model."]
    for p in prefixes:
        if key.startswith(p):
            return key[len(p):]
    return key

def main():
    # 0. 准备工作
    if not os.path.exists(PATH_INPUT_FOLDER):
        print(f"❌ 找不到输入文件夹: {PATH_INPUT_FOLDER}")
        return
    
    os.makedirs(PATH_OUTPUT_FOLDER, exist_ok=True)

    # 1. 加载 Delta (差值包)
    print(f"🔹 正在加载差值文件: {os.path.basename(PATH_DELTA)} ...")
    print("⏳ Delta 文件较大，加载需要一些时间，请耐心等待...")
    try:
        if PATH_DELTA.endswith(".safetensors"):
            delta_state = load_file(PATH_DELTA, device="cpu")
        else:
            delta_state = torch.load(PATH_DELTA, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"❌ 加载 Delta 失败: {e}")
        return

    # 建立 Delta 索引表 {CleanKey: RealKey}
    delta_map = {clean_key(k): k for k in delta_state.keys()}
    print(f"✅ Delta 加载完毕，包含 {len(delta_map)} 个层修改信息。")

    # 2. 扫描输入文件夹的分卷
    slice_files = sorted(glob.glob(os.path.join(PATH_INPUT_FOLDER, "*.safetensors")))
    if not slice_files:
        print(f"❌ 在 {PATH_INPUT_FOLDER} 没找到分卷文件！")
        return

    print(f"📦 检测到 {len(slice_files)} 个分卷文件，开始逐个处理...")

    total_modified = 0

    # 3. 逐个处理分卷
    for i, file_path in enumerate(slice_files):
        filename = os.path.basename(file_path)
        save_path = os.path.join(PATH_OUTPUT_FOLDER, filename)
        
        print(f"\n--- [分卷 {i+1}/{len(slice_files)}] 处理中: {filename} ---")
        
        # 加载分卷
        try:
            slice_state = load_file(file_path, device="cpu")
        except Exception as e:
            print(f"⚠️ 无法加载分卷 {filename}: {e}")
            continue

        modified_in_slice = 0
        new_slice_state = {}

        # 遍历当前分卷的每一层
        for key, tensor in tqdm(slice_state.items(), desc="注入进度"):
            clean_k = clean_key(key)
            
            # 检查 Delta 里有没有这一层
            if clean_k in delta_map:
                delta_key = delta_map[clean_k]
                delta_tensor = delta_state[delta_key]

                # 形状检查
                if tensor.shape == delta_tensor.shape:
                    # === 核心计算: 原参数 + (差值 * 强度) ===
                    # 转 float32 计算以防溢出，算完转回原格式(通常是bf16/fp16)
                    dtype_orig = tensor.dtype
                    new_tensor = tensor.to(torch.float32) + (delta_tensor.to(torch.float32) * ALPHA)
                    new_slice_state[key] = new_tensor.to(dtype_orig)
                    
                    modified_in_slice += 1
                else:
                    # 形状对不上，保留原样
                    new_slice_state[key] = tensor
            else:
                # Delta 里没有这一层，保留原样
                new_slice_state[key] = tensor

        total_modified += modified_in_slice
        
        # 保存修改后的分卷
        print(f"💾 保存分卷到: {save_path}")
        save_file(new_slice_state, save_path)

        # 清理内存
        del slice_state
        del new_slice_state
        gc.collect()

    # 4. 结束
    print("\n" + "="*40)
    print(f"🎉 处理完成！")
    print(f"📊 共修改了 {total_modified} 层")
    print(f"📂 新模型保存在: {PATH_OUTPUT_FOLDER}")
    print("👉 请进行下一步操作（如果是第一步，请继续运行第二步注入）。")

if __name__ == "__main__":
    main()