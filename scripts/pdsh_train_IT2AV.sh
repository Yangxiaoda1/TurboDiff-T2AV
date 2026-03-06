#!/bin/bash

# 多节点分布式训练启动脚本 (适配 TurboDiffusion + torchrun)
# 使用方法: bash scripts/pdsh_train_IT2AV.sh
# 修改下方 NODE_IP_LIST 以更换节点（强制覆盖环境变量，避免旧值 29.72.157.145）

export PDSH_RCMD_TYPE=ssh

# 强制使用当前 3 机 8 卡节点（不读取环境变量，避免 stale 的 29.72.157.145）
# NODE_IP_LIST="29.81.226.91,29.81.240.72,29.81.240.207"
# export NODE_IP_LIST

# 解析节点IP列表 (去除可能的端口号，将逗号替换为空格)
if [ -z "${NODE_IP_LIST}" ]; then
    echo "Error: NODE_IP_LIST environment variable is not set."
    exit 1
fi
node_ips=($(echo ${NODE_IP_LIST} | sed 's/:[0-9]*//g' | tr ',' ' '))

# 设置主节点
export MASTER_ADDR=${node_ips[0]}
echo "[pdsh_train] NODE_IP_LIST=$NODE_IP_LIST MASTER_ADDR=$MASTER_ADDR (若出现 29.72.157.145 请检查)"

# 获取可用端口函数
get_cluster_free_port() {
    local start_port=${1:-36000}
    local port=$start_port
    while true; do
        local check_output
        # 使用 bind 来检测端口是否真的可用 (比 connect 更准确，因为我们需要 bind)
        check_output=$(pdsh -w "$MASTER_ADDR" "python3 -c \"import socket as s; p=$port; so=s.socket(s.AF_INET, s.SOCK_STREAM); 
try:
    so.bind(('', p)); print('free')
except:
    print('used')
finally:
    so.close()\"" 2>/dev/null)
        
        # 必须明确匹配到 'free' 才认为可用
        if echo "$check_output" | grep -q "free"; then
            echo $port
            return 0
        fi
        port=$((port+1))
        if [ $port -gt 65000 ]; then port=36000; fi
    done
}

export MASTER_PORT=$(get_cluster_free_port 36000)
echo "[pdsh_train.sh] Using MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

# 项目根目录
REPO_ROOT="/apdcephfs_gy2/share_302507476/xiaodayang/AudioSummary/TurboDiffusion"
WORKDIR=${REPO_ROOT}

# 环境激活命令
ACTIVATE_CMD="source /apdcephfs_gy2/share_302507476/xiaodayang/miniconda3/bin/activate && conda activate turbodiff"

# 环境变量设置
    # 注意：WANDB_API_KEY 等敏感信息最好不要硬编码在脚本里，这里为了方便直接写入
# 12 机时 fsdp_shard_size=8 可减少跨节点通信（每节点 8 卡一组），比 96 全跨节点快很多
# 若显存不足可改为 16 或 24，但 12 机下会变慢
FSDP_SHARD_SIZE=8
# NCCL: 保持较短超时便于尽早发现死锁；若遇 collective timeout 多为 dataloader 步数不一致导致，已由 FixedEpochDataset 循环补齐修复
ENV_VARS="export PYTHONPATH=turbodiffusion && \
export NCCL_TIMEOUT=120 && \
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=120 && \
export NCCL_DEBUG=WARN && \
export NCCL_SOCKET_IFNAME=bond1 && \
export IMAGINAIRE_OUTPUT_ROOT="/apdcephfs_gy5/share_302507476/xiaodayang/AudioSummary/TurboDiffusion/ckpt_save100_193" && \
export MOVA_ROOT=\"/apdcephfs_gy2/share_302507476/xiaodayang/AudioSummary/MOVA\" && \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 && \
export WANDB_API_KEY=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 && \
export WANDB_ENTITY=1992426088-shanghai-ailab"

# 训练参数
NNODES=${#node_ips[@]}
NPROC_PER_NODE=8
TOTAL_GPUS=$((NNODES * NPROC_PER_NODE)) 

echo "[pdsh_train.sh] Launching on ${NNODES} nodes (${TOTAL_GPUS} GPUs total): ${node_ips[*]} (fsdp_shard_size=${FSDP_SHARD_SIZE})"

for i in "${!node_ips[@]}"; do
    ip=${node_ips[$i]}
    node_rank=$i
    
    echo "Launching on node ${ip} with rank ${node_rank}..."
    
    # 构建远程执行命令
    # 注意：
    # 1. cd 到项目根目录
    # 2. 激活环境
    # 3. 设置环境变量
    # 4. 运行 torchrun
    # 5. fsdp_shard_size=8：每节点 8 卡一组，减少跨节点 all-gather；96 会全跨节点，12 机时很慢
    pdsh -w "$ip" "cd ${REPO_ROOT} && \
        ${ACTIVATE_CMD} && \
        ${ENV_VARS} && \
        export http_proxy=http://star-proxy.oa.com:3128 && export https_proxy=http://star-proxy.oa.com:3128 && \
        export MASTER_ADDR=${MASTER_ADDR} && \
        export MASTER_PORT=${MASTER_PORT} && \
        torchrun --nnodes=${NNODES} --node_rank=${node_rank} --nproc_per_node=${NPROC_PER_NODE} \
          --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
          -m scripts.train \
          --config=turbodiffusion/rcm/configs/registry_rcm.py \
          -- \
          experiment=mova_360p_it2av_rcm \
          model=fsdp_it2av_distill_rcm \
          model.config.fsdp_shard_size=${FSDP_SHARD_SIZE} \
          model_parallel.context_parallel_size=8 \
          model.config.teacher_ckpt_path=\"/apdcephfs_gy2/share_302507476/xiaodayang/MOVA/MOVA-360p\" \
          model.config.student_ckpt_path=\"/apdcephfs_gy5/share_302507476/xiaodayang/AudioSummary/TurboDiffusion/ckpt_org/turbo_mova_video/turbo_t2av_core\" \
          dataloader_train.tar_path_pattern=\"/apdcephfs_gy5/share_302507476/xiaodayang/TurboDiffusion-Data/TrainPrepare/output_193/shard_*.tar\" \
          dataloader_train.num_workers=1 \
          dataloader_train.prefetch_factor=1 \
          model.config.use_gradient_checkpointing_offload=True \
          model.config.teacher_guidance=5.0 \
          checkpoint.save_iter=100 \
          optimizer=adamw" &
done

# 等待所有后台任务完成
wait
echo "[pdsh_train.sh] All training processes completed."
