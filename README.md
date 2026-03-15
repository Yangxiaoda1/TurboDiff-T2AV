# T2AV
## Setup
```
conda create -n turbodiff python=3.10.19
conda activate turbodiff
cd TurboDiffusion
git submodule update --init --recursive
pip install -e . --no-build-isolation
pip install git+https://github.com/thu-ml/SpargeAttn.git --no-build-isolation
pip install -r requirements.txt
```

## Preparation
### 导入库：MOVA和TurboDiffusion并列放
```
git clone https://github.com/OpenMOSS/MOVA.git
git clone https://github.com/Yangxiaoda1/TurboDiff-T2AV.git
```
### 下载权重：

TurboWan2.2-I2V-A14B-720P：https://huggingface.co/TurboDiffusion/TurboWan2.2-I2V-A14B-720P

T2AV as teacher model (MOVA): https://huggingface.co/OpenMOSS-Team/MOVA-360p

T2V-4-step-distilled as student model: https://huggingface.co/luyu1021/turbo_t2av/tree/main

### 数据
T2AV
原始格式：
```
|-1
    |-image.png
    |-prompt.txt
```
处理代码：
```
bash /apdcephfs_gy2/share_302507476/xiaodayang/AudioSummary/TurboDiffusion/turbodiffusion/rcm/datasets/build_synthetic_dataset_IT2AV_pdsh.sh
```
处理后格式（tar）：
```
.v_latent.pt：MOVA 生成的视频 VAE Latent (作为视频 x0)。
.a_latent.pt：MOVA 生成的音频 VAE Latent (作为音频 x0)。
.ref.pt ：参考图 (Ref Image)，它的Embedding，MOVA 依赖它做 Condition。
.embed.pt：文本 Embedding
.neg_embed.pt
.prompt.txt：原始文本
```


## IT2AV-Train(rCM):
```
bash /apdcephfs_gy2/share_302507476/xiaodayang/AudioSummary/TurboDiffusion/scripts/pdsh_train_IT2AV.sh
```

## DCP convert to .pt：
```
export PYTHONPATH=turbodiffusion
python scripts/convert_dcp_to_pt.py \
  "/apdcephfs_gy5/share_302507476/xiaodayang/AudioSummary/TurboDiffusion/ckpt_save100_193/rcm/RCM_MOVA/mova_360p_it2av_rcm/checkpoints/iter_000001600" \
  "/apdcephfs_gy5/share_302507476/xiaodayang/AudioSummary/TurboDiffusion/ckpt_save100_193/rcm/RCM_MOVA/mova_360p_it2av_rcm/checkpoints/student_iter_1600.pt"
```

## IT2AV-Infer(rCM):
```
bash /apdcephfs_gy2/share_302507476/xiaodayang/AudioSummary/TurboDiffusion/turbodiffusion/inference/run_mova_infer_IT2AV.sh
```

## T2A-Infer (MOVA Audio Branch Only)
```
bash /home/jovyan/codes/turbodiff/turbodiff_only_origin_t2a_infer/TurboDiff-T2AV/turbodiffusion/inference/run_mova_infer_T2A.sh
```

脚本：
`turbodiffusion/inference/run_mova_infer_T2A.sh`

说明：
1. 只使用 MOVA 原生音频分支推理，不依赖 student ckpt。
2. 两种输入模式，二选一：
   - 单条文本：`--prompt "..."`（或 `PROMPT="..."`）
   - 单个文本文件：`--prompt-file /path/prompts.txt`（每行一条 prompt）
3. 不做随机抽样，传入什么就跑什么。
4. 不传 `--prompt/--prompt-file` 时也合法：脚本会自动生成 `OUTPUT_DIR/default_prompt.txt`（单条默认 prompt）并执行推理。

常用环境变量：
`CKPT_PATH` `MOVA_ROOT` `PYTHON_BIN` `DEVICE` `OUTPUT_DIR` `NUM_INFERENCE_STEPS` `CFG_SCALE`

### 0) 无参数直接推理（合法）
```bash
bash /home/jovyan/codes/turbodiff/turbodiff_only_origin_t2a_infer/TurboDiff-T2AV/turbodiffusion/inference/run_mova_infer_T2A.sh
```
默认会在 `OUTPUT_DIR` 下创建 `default_prompt.txt`，并输出 `sample_00.wav`。

### 1) 单条文本推理
```bash
CKPT_PATH=/data/datasets/turbodiff_datasets_and_ckpt/MOVA_checkpoints/MOVA-360p \
MOVA_ROOT=/home/jovyan/codes/turbodiff/MOVA \
PYTHON_BIN=/home/jovyan/codes/turbodiff/TurboDiff-T2AV/.pixi/envs/default/bin/python \
DEVICE=cuda:0 \
bash turbodiffusion/inference/run_mova_infer_T2A.sh \
  --prompt "sing a song"
```
默认输出到：`$OUTPUT_DIR/output.wav`（若未设置 `OUTPUT_DIR`，默认是 `t2a_output/output.wav`）。

### 2) 单文件批量推理
```bash
CKPT_PATH=/data/datasets/turbodiff_datasets_and_ckpt/MOVA_checkpoints/MOVA-360p \
MOVA_ROOT=/home/jovyan/codes/turbodiff/MOVA \
PYTHON_BIN=/home/jovyan/codes/turbodiff/TurboDiff-T2AV/.pixi/envs/default/bin/python \
DEVICE=cuda:0 \
bash turbodiffusion/inference/run_mova_infer_T2A.sh \
  --prompt-file /home/jovyan/codes/turbodiff/turbodiff_only_origin_t2a_infer/TurboDiff-T2AV/input/prompts_1.txt \
  --output-dir /home/jovyan/codes/turbodiff/turbodiff_only_origin_t2a_infer/TurboDiff-T2AV/t2a_output
```
输出文件默认命名：`sample_00.wav`, `sample_01.wav`, ...，并在输出目录生成 `batch_prompts.txt`。

可选参数示例：
```bash
NUM_INFERENCE_STEPS=50 CFG_SCALE=5.0 SEED=42 SEED_BASE=1000 WRITE_PROMPT_TXT=1 \
bash turbodiffusion/inference/run_mova_infer_T2A.sh --prompt "sing a song"
```


## IT2AV-Train(SLA):
```
torchrun --nproc_per_node=8 \
  -m scripts.train --config="turbodiffusion/rcm/configs/${registry}.py" -- experiment="mova_360p_it2av_sla" \
  model=ddp_it2av_distill_sla \
  model.config.mova_ckpt_path="${MOVA_CKPT}" \
  dataloader_train.tar_path_pattern="${IT2AV_DATASET}/shard_*.tar" \
  model.config.student_num_layers_video=16 \
  model.config.student_num_layers_audio=16
  
```

## Tricks
TurboDiff-Wan+MOVA-Video
```
@宇翔代码
```
