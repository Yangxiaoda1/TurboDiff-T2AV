#!/bin/bash
# cd /home/jovyan/codes/turbodiff/new_Turbo
# bash TurboDiff-T2AV/scripts/train_t2a_single_node_8gpu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PIXI_ENV_ROOT="${PIXI_ENV_ROOT:-${REPO_ROOT}/.pixi/envs/default}"

export PYTHONPATH=turbodiffusion
export MOVA_ROOT="${MOVA_ROOT:-/home/jovyan/codes/turbodiff/new_Turbo/MOVA}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-${REPO_ROOT}/outputs_t2a}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-120}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-120}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

if [ -z "${VENV_SITE_PKGS:-}" ]; then
  if [ -d "${PIXI_ENV_ROOT}/lib/python3.12/site-packages" ]; then
    VENV_SITE_PKGS="${PIXI_ENV_ROOT}/lib/python3.12/site-packages"
  else
    VENV_SITE_PKGS="/home/jovyan/codes/turbodiff/new_Turbo/.venv/lib/python3.12/site-packages"
  fi
fi

if [ -f "${VENV_SITE_PKGS}/nvidia/cuda_runtime/lib/libcudart.so.12" ] && [ ! -e "${VENV_SITE_PKGS}/nvidia/cuda_runtime/lib/libcudart.so" ]; then
  ln -s libcudart.so.12 "${VENV_SITE_PKGS}/nvidia/cuda_runtime/lib/libcudart.so"
fi

TORCH_LIB_DIR="${TORCH_LIB_DIR:-${VENV_SITE_PKGS}/torch/lib}"
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${PIXI_ENV_ROOT}/lib:${VENV_SITE_PKGS}/nvidia/cuda_runtime/lib:${VENV_SITE_PKGS}/nvidia/cublas/lib:${VENV_SITE_PKGS}/nvidia/cudnn/lib:${VENV_SITE_PKGS}/nvidia/cufft/lib:${VENV_SITE_PKGS}/nvidia/curand/lib:${VENV_SITE_PKGS}/nvidia/cusolver/lib:${VENV_SITE_PKGS}/nvidia/cusparse/lib:${VENV_SITE_PKGS}/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "${PIXI_ENV_ROOT}/bin/python" ]; then
    PYTHON_BIN="${PIXI_ENV_ROOT}/bin/python"
  elif [ -x "/home/jovyan/codes/turbodiff/new_Turbo/.venv/bin/python" ]; then
    PYTHON_BIN="/home/jovyan/codes/turbodiff/new_Turbo/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[train_t2a] PYTHON_BIN not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [ ! -d "${VENV_SITE_PKGS}" ]; then
  echo "[train_t2a] site-packages not found: ${VENV_SITE_PKGS}" >&2
  exit 1
fi

if [ ! -d "${TORCH_LIB_DIR}" ]; then
  echo "[train_t2a] torch lib dir not found: ${TORCH_LIB_DIR}" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import sys
import torch

print(sys.executable)
print(torch.__version__)
PY
then
  echo "[train_t2a] Failed to import torch with current runtime libraries." >&2
  echo "[train_t2a] Check that the pixi Python env is active and torch can load CUDA dependencies." >&2
  echo "[train_t2a] PYTHON_BIN=${PYTHON_BIN}" >&2
  echo "[train_t2a] VENV_SITE_PKGS=${VENV_SITE_PKGS}" >&2
  echo "[train_t2a] TORCH_LIB_DIR=${TORCH_LIB_DIR}" >&2
  exit 1
fi

TEACHER_CKPT_PATH="${TEACHER_CKPT_PATH:-/data/datasets/turbodiff_datasets_and_ckpt/MOVA_checkpoints/MOVA-360p}"
STUDENT_CKPT_PATH="${STUDENT_CKPT_PATH:-${TEACHER_CKPT_PATH}}"
T2A_DATASET_ROOT="${T2A_DATASET_ROOT:-/data/datasets/turbodiff_datasets_and_ckpt/t2a_latents}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29600}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
REGISTRY="${REGISTRY:-registry_rcm}"
EXPERIMENT="${EXPERIMENT:-mova_360p_t2a_rcm}"
MODEL_NAME="${MODEL_NAME:-ddp_t2a_distill_rcm}"
OPTIMIZER_NAME="${OPTIMIZER_NAME:-}"
STUDENT_NUM_LAYERS_AUDIO="${STUDENT_NUM_LAYERS_AUDIO:-0}"
STUDENT_UPDATE_FREQ="${STUDENT_UPDATE_FREQ:-5}"
ITERATION_OFFSET="${ITERATION_OFFSET:-0}"
TANGENT_WARMUP="${TANGENT_WARMUP:-1000}"
TEACHER_GUIDANCE="${TEACHER_GUIDANCE:-5.0}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-noise, distorted, low quality, bad audio, muffled, static}"
LOSS_SCALE_DMD="${LOSS_SCALE_DMD:-}"
LOSS_SCALE_FAKE_SCORE="${LOSS_SCALE_FAKE_SCORE:-}"
LOSS_SCALE_TEACHER="${LOSS_SCALE_TEACHER:-}"
FAKE_SCORE_LR="${FAKE_SCORE_LR:-}"
SAVE_ITER="${SAVE_ITER:-200}"
TRAIN_MAX_ITER="${TRAIN_MAX_ITER:-100000}"
BATCH_INFER_ENABLED="${BATCH_INFER_ENABLED:-1}"
BATCH_INFER_EVERY="${BATCH_INFER_EVERY:-5000}"
BATCH_INFER_POLL_SECONDS="${BATCH_INFER_POLL_SECONDS:-30}"
BATCH_INFER_DEVICE="${BATCH_INFER_DEVICE:-cuda:0}"
BATCH_INFER_MAX_PROMPTS="${BATCH_INFER_MAX_PROMPTS:-10}"
JOB_PROJECT="${JOB_PROJECT:-rcm}"
JOB_GROUP="${JOB_GROUP:-RCM_MOVA}"
JOB_NAME="${JOB_NAME:-${EXPERIMENT}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${IMAGINAIRE_OUTPUT_ROOT}/${JOB_PROJECT}/${JOB_GROUP}/${JOB_NAME}/checkpoints}"
BATCH_INFER_OUTPUT_ROOT="${BATCH_INFER_OUTPUT_ROOT:-${IMAGINAIRE_OUTPUT_ROOT}/${JOB_PROJECT}/${JOB_GROUP}/${JOB_NAME}/batch_infer}"
NEGATIVE_PROMPT_ESCAPED="${NEGATIVE_PROMPT//\'/\\\'}"
printf -v NEGATIVE_PROMPT_OVERRIDE "model.config.negative_prompt='%s'" "${NEGATIVE_PROMPT_ESCAPED}"

if [ -z "${PROMPT_SOURCE:-}" ]; then
  PROMPT_SOURCE="/data/datasets/turbodiff_datasets_and_ckpt/wan_audio/wanaudio_prompts.txt"
fi

SAVE_ITER=1000

if [ -z "${OPTIMIZER_NAME}" ]; then
  if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("transformer_engine") is not None else 1)
PY
  then
    OPTIMIZER_NAME="fusedadamw"
  else
    OPTIMIZER_NAME="adamw"
    echo "[train_t2a] transformer_engine not found; falling back to adamw while keeping t2v-aligned betas from the experiment config."
  fi
fi

cd "${REPO_ROOT}"

echo "[train_t2a] Launch single-node ${NPROC_PER_NODE} GPUs"
echo "[train_t2a] PIXI_ENV_ROOT=${PIXI_ENV_ROOT}"
echo "[train_t2a] PYTHON_BIN=${PYTHON_BIN}"
echo "[train_t2a] VENV_SITE_PKGS=${VENV_SITE_PKGS}"
echo "[train_t2a] TORCH_LIB_DIR=${TORCH_LIB_DIR}"
echo "[train_t2a] MOVA_ROOT=${MOVA_ROOT}"
echo "[train_t2a] TEACHER_CKPT_PATH=${TEACHER_CKPT_PATH}"
echo "[train_t2a] STUDENT_CKPT_PATH=${STUDENT_CKPT_PATH}"
echo "[train_t2a] T2A_DATASET_ROOT=${T2A_DATASET_ROOT}"
echo "[train_t2a] REGISTRY=${REGISTRY} EXPERIMENT=${EXPERIMENT} MODEL_NAME=${MODEL_NAME} OPTIMIZER_NAME=${OPTIMIZER_NAME}"
echo "[train_t2a] STUDENT_NUM_LAYERS_AUDIO=${STUDENT_NUM_LAYERS_AUDIO} STUDENT_UPDATE_FREQ=${STUDENT_UPDATE_FREQ} ITERATION_OFFSET=${ITERATION_OFFSET} TANGENT_WARMUP=${TANGENT_WARMUP} TEACHER_GUIDANCE=${TEACHER_GUIDANCE}"
echo "[train_t2a] NEGATIVE_PROMPT=${NEGATIVE_PROMPT}"
echo "[train_t2a] CHECKPOINT_DIR=${CHECKPOINT_DIR}"
if [ "${BATCH_INFER_ENABLED}" = "1" ]; then
  echo "[train_t2a] batch inference every ${BATCH_INFER_EVERY} iters -> ${BATCH_INFER_OUTPUT_ROOT}"
fi
if [ -n "${LOSS_SCALE_DMD}" ]; then
  echo "[train_t2a] LOSS_SCALE_DMD=${LOSS_SCALE_DMD}"
fi
if [ -n "${LOSS_SCALE_FAKE_SCORE}" ]; then
  echo "[train_t2a] LOSS_SCALE_FAKE_SCORE=${LOSS_SCALE_FAKE_SCORE}"
fi
if [ -n "${LOSS_SCALE_TEACHER}" ]; then
  echo "[train_t2a] LOSS_SCALE_TEACHER=${LOSS_SCALE_TEACHER}"
fi
if [ -n "${FAKE_SCORE_LR}" ]; then
  echo "[train_t2a] FAKE_SCORE_LR=${FAKE_SCORE_LR}"
fi

EXTRA_ARGS=()
if [ -n "${LOSS_SCALE_DMD}" ]; then
  EXTRA_ARGS+=("model.config.loss_scale_dmd=${LOSS_SCALE_DMD}")
fi
if [ -n "${LOSS_SCALE_FAKE_SCORE}" ]; then
  EXTRA_ARGS+=("model.config.loss_scale_fake_score=${LOSS_SCALE_FAKE_SCORE}")
fi
if [ -n "${LOSS_SCALE_TEACHER}" ]; then
  EXTRA_ARGS+=("model.config.loss_scale_teacher=${LOSS_SCALE_TEACHER}")
fi
if [ -n "${FAKE_SCORE_LR}" ]; then
  EXTRA_ARGS+=("model.config.fake_score_lr=${FAKE_SCORE_LR}")
fi

run_batch_inference_for_iter() {
  local iteration="$1"
  local prompt_file="$2"
  local infer_output_dir="${BATCH_INFER_OUTPUT_ROOT}/iter_$(printf '%09d' "${iteration}")"
  local student_ckpt_path="${CHECKPOINT_DIR}/iter_$(printf '%09d' "${iteration}")"
  local done_file="${infer_output_dir}/.done"

  if [ ! -e "${student_ckpt_path}" ]; then
    echo "[train_t2a] Skip batch inference for iter ${iteration}: checkpoint missing at ${student_ckpt_path}"
    return 0
  fi

  if [ -e "${done_file}" ]; then
    return 0
  fi

  mkdir -p "${infer_output_dir}"

  echo "[train_t2a] Running batch inference for iter ${iteration}"

  local inference_script="${SCRIPT_DIR}/inference_mova_t2a.sh"
  local infer_specs=(
    "teacher:50"
    "teacher:10"
    "teacher:4"
    "student:10"
    "student:4"
  )
  local spec
  for spec in "${infer_specs[@]}"; do
    local mode="${spec%%:*}"
    local steps="${spec##*:}"
    PIXI_ENV_ROOT="${PIXI_ENV_ROOT}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    MOVA_ROOT="${MOVA_ROOT}" \
    TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE}" \
    CKPT_PATH="${TEACHER_CKPT_PATH}" \
    STUDENT_CKPT_PATH="${student_ckpt_path}" \
    INFER_MODE="${mode}" \
    NUM_INFERENCE_STEPS="${steps}" \
    OUTPUT_DIR="${infer_output_dir}" \
    PROMPT_FILE="${prompt_file}" \
    GENERATE_PROMPT_FILE=0 \
    PROMPT_SOURCE="${PROMPT_SOURCE}" \
    BATCH_MAX_PROMPTS="${BATCH_INFER_MAX_PROMPTS}" \
    DEVICE="${BATCH_INFER_DEVICE}" \
    NEGATIVE_PROMPT="${NEGATIVE_PROMPT}" \
    bash "${inference_script}"
  done

  touch "${done_file}"
}

watch_checkpoints_and_infer() {
  local prompt_file="${BATCH_INFER_OUTPUT_ROOT}/shared_prompts.txt"
  local min_iteration=0
  mkdir -p "${BATCH_INFER_OUTPUT_ROOT}"
  if [ ! -f "${prompt_file}" ]; then
    shuf -n "${BATCH_INFER_MAX_PROMPTS}" "${PROMPT_SOURCE}" > "${prompt_file}"
  fi
  if [ -d "${CHECKPOINT_DIR}" ]; then
    local existing_checkpoint
    for existing_checkpoint in "${CHECKPOINT_DIR}"/iter_*; do
      [ -e "${existing_checkpoint}" ] || continue
      local existing_name="$(basename "${existing_checkpoint}")"
      local existing_iteration="${existing_name#iter_}"
      existing_iteration="$((10#${existing_iteration}))"
      if [ "${existing_iteration}" -gt "${min_iteration}" ]; then
        min_iteration="${existing_iteration}"
      fi
    done
  fi

  echo "[train_t2a] Watcher started: polling ${CHECKPOINT_DIR} every ${BATCH_INFER_POLL_SECONDS}s"
  if [ "${min_iteration}" -gt 0 ]; then
    echo "[train_t2a] Watcher will ignore existing checkpoints up to iter ${min_iteration}"
  fi

  while kill -0 "${TRAIN_PID}" >/dev/null 2>&1; do
    if [ -d "${CHECKPOINT_DIR}" ]; then
      local checkpoint_path
      for checkpoint_path in "${CHECKPOINT_DIR}"/iter_*; do
        [ -e "${checkpoint_path}" ] || continue
        local checkpoint_name="$(basename "${checkpoint_path}")"
        local iteration="${checkpoint_name#iter_}"
        iteration="$((10#${iteration}))"
        if [ "${iteration}" -le "${min_iteration}" ]; then
          continue
        fi
        if [ $((iteration % BATCH_INFER_EVERY)) -eq 0 ]; then
          run_batch_inference_for_iter "${iteration}" "${prompt_file}"
        fi
      done
    fi
    sleep "${BATCH_INFER_POLL_SECONDS}"
  done
}

TRAIN_ARGS=(
  -m torch.distributed.run
  --nnodes=1
  --node_rank=0
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_addr="${MASTER_ADDR}"
  --master_port="${MASTER_PORT}"
  -m scripts.train
  --config="turbodiffusion/rcm/configs/${REGISTRY}.py"
  --
  experiment="${EXPERIMENT}"
  model="${MODEL_NAME}"
  optimizer="${OPTIMIZER_NAME}"
  model.config.teacher_ckpt_path="${TEACHER_CKPT_PATH}"
  model.config.student_ckpt_path="${STUDENT_CKPT_PATH}"
  model.config.student_num_layers_audio="${STUDENT_NUM_LAYERS_AUDIO}"
  model.config.student_update_freq="${STUDENT_UPDATE_FREQ}"
  model.config.iteration_offset="${ITERATION_OFFSET}"
  model.config.tangent_warmup="${TANGENT_WARMUP}"
  model.config.teacher_guidance="${TEACHER_GUIDANCE}"
  "${NEGATIVE_PROMPT_OVERRIDE}"
  dataloader_train.tar_path_pattern="${T2A_DATASET_ROOT}/shard_*.tar"
  dataloader_train.batch_size=2
  dataloader_train.num_workers=1
  dataloader_train.prefetch_factor=1
  trainer.max_iter="${TRAIN_MAX_ITER}"
  checkpoint.save_iter="${SAVE_ITER}"
  job.wandb_mode=disabled
  "${EXTRA_ARGS[@]}"
)

cleanup() {
  if [ -n "${WATCHER_PID:-}" ] && kill -0 "${WATCHER_PID}" >/dev/null 2>&1; then
    kill "${WATCHER_PID}" >/dev/null 2>&1 || true
    wait "${WATCHER_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT

"${PYTHON_BIN}" "${TRAIN_ARGS[@]}" &
TRAIN_PID=$!

if [ "${BATCH_INFER_ENABLED}" = "1" ]; then
  watch_checkpoints_and_infer &
  WATCHER_PID=$!
fi

set +e
wait "${TRAIN_PID}"
TRAIN_EXIT_CODE=$?
set -e

cleanup

exit "${TRAIN_EXIT_CODE}"
