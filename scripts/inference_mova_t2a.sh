#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PIXI_ENV_ROOT="${PIXI_ENV_ROOT:-${REPO_ROOT}/.pixi/envs/default}"
PY_SCRIPT="${REPO_ROOT}/turbodiffusion/inference/mova_t2a_infer.py"

export MOVA_ROOT="${MOVA_ROOT:-/home/jovyan/codes/turbodiff/new_Turbo/MOVA}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "${PIXI_ENV_ROOT}/bin/python" ]; then
    PYTHON_BIN="${PIXI_ENV_ROOT}/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

CKPT_PATH="${CKPT_PATH:-/data/datasets/turbodiff_datasets_and_ckpt/MOVA_checkpoints/MOVA-360p}"
STUDENT_CKPT_NUM="${STUDENT_CKPT_NUM:-2000}"
printf -v STUDENT_CKPT_NUM_PADDED "%09d" "${STUDENT_CKPT_NUM}"
STUDENT_CKPT_PATH="${STUDENT_CKPT_PATH:-${REPO_ROOT}/outputs_t2a/rcm/RCM_MOVA/mova_360p_t2a_rcm/checkpoints/iter_${STUDENT_CKPT_NUM_PADDED}}"
DEVICE="${DEVICE:-cuda:0}"
INFER_MODE="${INFER_MODE:-all}"
TEACHER_CFG_SCALE="${TEACHER_CFG_SCALE:-5.0}"
STUDENT_CFG_SCALE="${STUDENT_CFG_SCALE:-5.0}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-noise, distorted, low quality, bad audio, muffled, static}"
NUM_FRAMES="${NUM_FRAMES:-193}"
VIDEO_FPS="${VIDEO_FPS:-24.0}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs_t2a/rcm/RCM_MOVA/random_batch}"
SEED_BASE="${SEED_BASE:-42}"
SAVE_PREFIX="${SAVE_PREFIX:-sample_}"
WRITE_PROMPT_TXT="${WRITE_PROMPT_TXT:-1}"
BATCH_MAX_PROMPTS="${BATCH_MAX_PROMPTS:-10}"
PROMPT_SOURCE="${PROMPT_SOURCE:-/data/datasets/turbodiff_datasets_and_ckpt/wan_audio/wanaudio_prompts.txt}"
PROMPT_FILE="${PROMPT_FILE:-/tmp/random_prompts.txt}"
GENERATE_PROMPT_FILE="${GENERATE_PROMPT_FILE:-1}"

if [ ! -f "${PY_SCRIPT}" ]; then
  echo "[infer_t2a] inference entrypoint not found: ${PY_SCRIPT}" >&2
  exit 1
fi

if [ "${GENERATE_PROMPT_FILE}" = "1" ]; then
  mkdir -p "$(dirname "${PROMPT_FILE}")"
  shuf -n "${BATCH_MAX_PROMPTS}" "${PROMPT_SOURCE}" > "${PROMPT_FILE}"
fi

run_once_batch() {
  local mode="$1"
  local steps="$2"
  local tag="$3"
  local cfg_scale="$4"
  local tagged_output_dir="${OUTPUT_DIR}/${tag}"
  local tagged_prefix="${tag}_${SAVE_PREFIX}"

  echo "[infer_t2a] run=${tag} mode=${mode} steps=${steps} cfg_scale=${cfg_scale}"
  echo "[infer_t2a] output_dir=${tagged_output_dir}"

  local infer_args=(
    --ckpt_path "${CKPT_PATH}"
    --prompt_file "${PROMPT_FILE}"
    --output_dir "${tagged_output_dir}"
    --save_prefix "${tagged_prefix}"
    --seed_base "${SEED_BASE}"
    --negative_prompt "${NEGATIVE_PROMPT}"
    --num_inference_steps "${steps}"
    --cfg_scale "${cfg_scale}"
    --num_frames "${NUM_FRAMES}"
    --video_fps "${VIDEO_FPS}"
    --device "${DEVICE}"
  )

  if [ "${WRITE_PROMPT_TXT}" = "1" ]; then
    infer_args+=(--write_prompt_txt)
  fi

  if [ "${mode}" != "teacher" ]; then
    infer_args+=(--student_ckpt_path "${STUDENT_CKPT_PATH}")
  fi

  "${PYTHON_BIN}" "${PY_SCRIPT}" "${infer_args[@]}"
}

echo "[infer_t2a] Batch mode"
echo "[infer_t2a] prompt_file=${PROMPT_FILE}"
echo "[infer_t2a] output_dir=${OUTPUT_DIR}"
echo "[infer_t2a] batch_max_prompts=${BATCH_MAX_PROMPTS}"
echo "[infer_t2a] negative_prompt=${NEGATIVE_PROMPT}"
echo "[infer_t2a] teacher_cfg_scale=${TEACHER_CFG_SCALE} student_cfg_scale=${STUDENT_CFG_SCALE}"

if [ "${INFER_MODE}" = "all" ]; then
  run_once_batch teacher 50 teacher_50 "${TEACHER_CFG_SCALE}"
  run_once_batch teacher 10 teacher_10 "${TEACHER_CFG_SCALE}"
  run_once_batch teacher 4 teacher_4 "${TEACHER_CFG_SCALE}"
  run_once_batch student 10 student_10 "${STUDENT_CFG_SCALE}"
  run_once_batch student 4 student_4 "${STUDENT_CFG_SCALE}"
elif [ "${INFER_MODE}" = "teacher" ]; then
  NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
  run_once_batch teacher "${NUM_INFERENCE_STEPS}" "teacher_${NUM_INFERENCE_STEPS}" "${TEACHER_CFG_SCALE}"
else
  NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-4}"
  run_once_batch student "${NUM_INFERENCE_STEPS}" "student_${NUM_INFERENCE_STEPS}" "${STUDENT_CFG_SCALE}"
fi
