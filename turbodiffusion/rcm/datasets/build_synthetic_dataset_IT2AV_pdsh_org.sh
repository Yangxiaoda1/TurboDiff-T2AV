#!/bin/bash

# 多节点分布式数据生成启动脚本
# 使用方法: 
# 1. 确保环境变量 NODE_IP_LIST 已设置 (例如: "10.0.0.1,10.0.0.2")
# 2. 运行: bash turbodiffusion/rcm/datasets/build_synthetic_dataset_IT2AV_pdsh.sh

export PDSH_RCMD_TYPE=ssh

# 解析节点IP列表 (去除可能的端口号，将逗号替换为空格)
if [ -z "${NODE_IP_LIST}" ]; then
    echo "Error: NODE_IP_LIST environment variable is not set."
    exit 1
fi
node_ips=($(echo ${NODE_IP_LIST} | sed 's/:8//g' | tr ',' ' '))

# 设置主节点
export MASTER_ADDR=${node_ips[0]}

# 获取可用端口函数
get_cluster_free_port() {
    local start_port=${1:-36000}
    local port=$start_port
    while true; do
        local check_output
        # 使用 bind 来检测端口是否真的可用
        check_output=$(pdsh -w "$MASTER_ADDR" "python3 -c \"import socket as s; p=$port; so=s.socket(s.AF_INET, s.SOCK_STREAM); 
try:
    so.bind(('', p)); print('free')
except:
    print('used')
finally:
    so.close()\"" 2>/dev/null)
        
        if echo "$check_output" | grep -q "free"; then
            echo $port
            return 0
        fi
        port=$((port+1))
        if [ $port -gt 65000 ]; then port=36000; fi
    done
}

export MASTER_PORT=$(get_cluster_free_port 36000)
echo "[build_dataset] Using MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"

# 项目根目录
REPO_ROOT="/apdcephfs_gy2/share_302507476/xiaodayang/AudioSummary/TurboDiffusion"
WORKDIR=${REPO_ROOT}

# 环境激活命令
ACTIVATE_CMD="source /apdcephfs_gy2/share_302507476/xiaodayang/miniconda3/bin/activate && conda activate turbodiff"

# 环境变量设置
ENV_VARS="export PYTHONPATH=turbodiffusion && \
export TORCH_DISTRIBUTED_DEBUG=DETAIL && \
export NCCL_DEBUG=INFO && \
export NCCL_ASYNC_ERROR_HANDLING=1"

# 分布式参数
NNODES=${#node_ips[@]}
NPROC_PER_NODE=8
TOTAL_GPUS=$((NNODES * NPROC_PER_NODE))

echo "[build_dataset] Launching on ${NNODES} nodes (${TOTAL_GPUS} GPUs total): ${node_ips[*]}"

for i in "${!node_ips[@]}"; do
    ip=${node_ips[$i]}
    node_rank=$i
    
    echo "Launching on node ${ip} with rank ${node_rank}..."
    
    # 构建远程执行命令
    pdsh -w "$ip" "cd ${REPO_ROOT} && \
        ${ACTIVATE_CMD} && \
        ${ENV_VARS} && \
        export http_proxy=http://star-proxy.oa.com:3128 && export https_proxy=http://star-proxy.oa.com:3128 && \
        export MASTER_ADDR=${MASTER_ADDR} && \
        export MASTER_PORT=${MASTER_PORT} && \
        torchrun --nnodes=${NNODES} --node_rank=${node_rank} --nproc_per_node=${NPROC_PER_NODE} \
          --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
          /apdcephfs_gy2/share_302507476/xiaodayang/AudioSummary/TurboDiffusion/turbodiffusion/rcm/datasets/build_synthetic_dataset_IT2AV.py \
          --input_dir \"/apdcephfs_gy5/share_302507476/xiaodayang/TurboDiffusion-Data/TrainPrepare/input\" \
          --output_dir \"/apdcephfs_gy5/share_302507476/xiaodayang/TurboDiffusion-Data/TrainPrepare/output_193\" \
          --ckpt_path \"/apdcephfs_gy2/share_302507476/xiaodayang/MOVA/MOVA-360p\" \
          --height 352 \
          --width 640 \
          --num_frames 193 \
          --samples_per_shard 16" &
done

# 等待所有后台任务完成
wait
echo "[build_dataset] All processes completed."
