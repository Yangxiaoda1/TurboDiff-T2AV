import torch
import os
import glob
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import gc

# ================= ⚙️ 配置区域 =================

# 1. Turbo 模型路径 (被减数)
# PATH_TURBO = r"D:\xunleidownload\TurboWan2.2-I2V-A14B-high-720P.pth"
PATH_TURBO = r"D:\Models\TurboWan2.2-I2V-A14B-low-720P.pth"

# 2. Base 基础模型文件夹 (减数)
# PATH_BASE_FOLDER = r"D:\tju\Junior\zju-lab\Wan2.2-I2V-A14B\high_noise_model"
PATH_BASE_FOLDER = r"D:\Models\wan2.2-i2v-a14b-low-noise-model"

# 3. 输出保存路径
# 注意：确保 D 盘有至少 30GB 剩余空间
# SAVE_PATH = r"D:\Models\Wan2.2_14B_Turbo_Delta_High.safetensors"
SAVE_PATH = r"D:\Models\Wan2.2_14B_Turbo_Delta_Low.safetensors"

# ===============================================

def clean_key(key):
    """清洗 Key，去除常见前缀以增加匹配率"""
    prefixes = ["model.diffusion_model.", "diffusion_model.", "module.", "model."]
    for p in prefixes:
        if key.startswith(p):
            return key[len(p):]
    return key

def load_turbo_model(path):
    """加载 Turbo 模型"""
    print(f"📂 正在加载 Turbo 模型: {os.path.basename(path)} ...")
    print("⏳这可能需要几分钟，请耐心等待 (约 28GB 内存加载)...")
    
    if path.endswith(".safetensors"):
        return load_file(path, device="cpu")
    else:
        # 添加 weights_only=False 消除警告
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            # 尝试获取真正的权重字典
            return ckpt.get("state_dict", ckpt.get("model", ckpt.get("model_ema", ckpt)))
        return ckpt

def main():
    # 0. 检查路径
    if not os.path.exists(PATH_TURBO):
        print(f"❌ 错误：找不到 Turbo 文件: {PATH_TURBO}")
        return
    
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # 1. 加载 Turbo 模型 (内存占用峰值 1)
    try:
        turbo_raw = load_turbo_model(PATH_TURBO)
    except Exception as e:
        print(f"❌ 加载 Turbo 模型失败 (可能是内存不足): {e}")
        return
        
    print(f"✅ Turbo 加载完毕，共 {len(turbo_raw)} 层")

    # 建立映射表： {CleanKey: OriginalTurboKey}
    # 这一步是为了快速查找，不用每次都遍历
    turbo_map = {clean_key(k): k for k in turbo_raw.keys()}
    
    # 2. 扫描 Base 分卷
    base_files = sorted(glob.glob(os.path.join(PATH_BASE_FOLDER, "*.safetensors")))
    if not base_files:
        print(f"❌ 错误：在 {PATH_BASE_FOLDER} 没找到 .safetensors 文件")
        return
        
    print(f"📦 检测到 Base 模型包含 {len(base_files)} 个分卷文件")

    delta_state = {} # 用于存放最终结果
    processed_count = 0

    # 3. 流式遍历 Base 分卷
    print("🚀 开始计算差值 (Turbo - Base)...")
    
    for i, file_path in enumerate(base_files):
        print(f"\n--- [分卷 {i+1}/{len(base_files)}] 处理中: {os.path.basename(file_path)} ---")
        
        # 加载 Base 分卷
        try:
            base_shard = load_file(file_path, device="cpu")
        except Exception as e:
            print(f"⚠️ 无法加载分卷 {file_path}: {e}")
            continue

        # 遍历分卷中的每一层
        keys_to_remove_from_turbo = [] # 记录需要从 Turbo 中删除的 key

        for base_k, base_t in tqdm(base_shard.items(), desc="层处理进度"):
            clean_base_k = clean_key(base_k)
            
            # 匹配检查
            if clean_base_k in turbo_map:
                turbo_full_k = turbo_map[clean_base_k]
                
                # 再次确认 key 还在内存里 (可能之前已经被处理过了)
                if turbo_full_k not in turbo_raw:
                    continue

                turbo_t = turbo_raw[turbo_full_k]
                
                # 形状检查
                if base_t.shape != turbo_t.shape:
                    continue
                
                # === 核心计算 ===
                try:
                    # 1. 计算差值 (转 float32 保证精度)
                    delta = turbo_t.to(torch.float32) - base_t.to(torch.float32)
                    
                    # 2. 存入结果 (转 float16 节省内存)
                    delta_state[clean_base_k] = delta.to(torch.float16)
                    
                    # 3. 【关键优化】标记删除 Turbo 中的原数据
                    # 我们已经算出了差值，就不再需要原始的 Turbo 这一层数据了
                    # 这样可以释放内存给后续的 Delta 结果腾地方
                    keys_to_remove_from_turbo.append(turbo_full_k)
                    
                    processed_count += 1
                except Exception as e:
                    print(f"计算错误 {clean_base_k}: {e}")

        # === 内存释放阶段 ===
        
        # 1. 删除已使用的 Turbo 原始权重 (腾笼换鸟)
        for k in keys_to_remove_from_turbo:
            del turbo_raw[k]
        
        # 2. 删除当前的 Base 分卷
        del base_shard
        
        # 3. 强制 GC
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"♻️ 内存优化完成。剩余待处理 Turbo 层数: {len(turbo_raw)}")

    # 4. 保存结果
    print("\n" + "="*40)
    print(f"📊 最终报告:")
    print(f"✅ 成功提取并计算层数: {processed_count}")
    
    if processed_count == 0:
        print("❌ 失败：没有匹配到任何层，请检查模型版本是否对应。")
    else:
        print(f"💾 正在保存结果到硬盘 (请勿关闭)...")
        print(f"📂 路径: {SAVE_PATH}")
        try:
            save_file(delta_state, SAVE_PATH)
            print("🎉 恭喜！差值提取成功！")
        except Exception as e:
            print(f"❌ 保存文件失败: {e}")
            print("💡 请检查磁盘空间是否足够 (需要约 28GB)。")

if __name__ == "__main__":
    main()