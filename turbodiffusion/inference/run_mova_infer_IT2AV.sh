#!/bin/bash

# T2AV 推理启动脚本 (8卡并行)
# 运行前请确保已安装 ffmpeg (用于音视频合并)

export PYTHONPATH=turbodiffusion

# 原始 MOVA 权重路径
CKPT_PATH="/apdcephfs_gy2/share_302507476/xiaodayang/MOVA/MOVA-360p"

# 蒸馏后的 Student 权重路径 (可选)
# 请先运行 scripts/convert_dcp_to_pt.py 将 FSDP 权重转换为单文件
STUDENT_CKPT_PATH="/apdcephfs_gy5/share_302507476/xiaodayang/AudioSummary/TurboDiffusion/ckpt_save100_193/rcm/RCM_MOVA/mova_360p_it2av_rcm/checkpoints/student_iter_1600.pt" 

# IMAGE_PATH="assets/i2v_inputs/sailor.jpg"
IMAGE_PATH="/apdcephfs_gy2/share_302507476/xiaodayang/AudioSummary/MOVA/assets/single_person.jpg"
# 示例 Prompt
# PROMPT="Close-up on an elderly sailor in a weathered yellow raincoat, seated on the sun-lit deck of a gently rocking catamaran. With each small rise and dip of the hull, the shadows on his face shift subtly. He draws from his pipe, the ember brightening, and the exhale sends a thin ribbon of smoke that wavers and bends as the boat sways. His cat lies beside him, eyes half-closed, its body adjusting with soft, instinctive shifts whenever the deck tilts. Sunlight glints off the polished wood in flickering patterns as the surface of the water rolls beneath, causing brief flares of moving reflections. A pair of seabirds glide overhead, dipping slightly as the wind gusts. As the camera eases into a slow push-in, every motion becomes more pronounced—the smoke trembling, the cat’s fur fluttering, the deck creaking with each gentle sway—turning the peaceful moment into a dynamically living scene afloat at sea."
PROMPT="A man in a blue blazer and glasses speaks in a formal indoor setting, framed by wooden furniture and a filled bookshelf. Quiet room acoustics underscore his measured tone as he delivers his remarks. At one point, he says, \"I would also say that this election in Germany wasn’t surprising.\""

OUTPUT_DIR="/apdcephfs_gy2/share_302507476/xiaodayang/AudioSummary/TurboDiffusion/infer_output"
mkdir -p $OUTPUT_DIR

echo "Starting 1-GPU inference..."

for i in 0; do
    SAVE_PATH="${OUTPUT_DIR}/step4_cfg1_193_person_1600_${i}.mp4"
    SEED=$((42 + i))
    
    # 可以在这里为每个 GPU 设置不同的 Prompt 或 Image
    
    nohup python turbodiffusion/inference/mova_it2av_infer.py \
        --device_id $i \
        --ckpt_path "$CKPT_PATH" \
        --student_ckpt_path "$STUDENT_CKPT_PATH" \
        --image_path "$IMAGE_PATH" \
        --prompt "$PROMPT" \
        --save_path "$SAVE_PATH" \
        --steps 4 \
        --cfg_scale 1.0 \
        --seed $SEED \
        --height 352 \
        --width 640 \
        --num_frames 193 > "${OUTPUT_DIR}/log_gpu${i}.txt" 2>&1 &
        
    echo "Launched GPU $i, log: ${OUTPUT_DIR}/log_gpu${i}.txt"
done

wait
echo "All inference processes completed."
