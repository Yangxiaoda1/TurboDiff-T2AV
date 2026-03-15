#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PY_SCRIPT="${REPO_ROOT}/turbodiffusion/inference/mova_t2a_infer.py"
MOVA_ROOT="${MOVA_ROOT:-/home/jovyan/codes/turbodiff/MOVA}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/codes/turbodiff/TurboDiff-T2AV/.pixi/envs/default/bin/python}"

CKPT_PATH="${CKPT_PATH:-/data/datasets/turbodiff_datasets_and_ckpt/MOVA_checkpoints/MOVA-360p}"
STUDENT_CKPT_PATH="${STUDENT_CKPT_PATH:-}"
PROMPT_FILE="${PROMPT_FILE:-/data/datasets/turbodiff_datasets_and_ckpt/wan_audio/wanaudio_prompts.txt}"
SELECTED_PROMPTS_FILE="${SELECTED_PROMPTS_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/t2a_random10_out}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-5.0}"
NUM_FRAMES="${NUM_FRAMES:-193}"
VIDEO_FPS="${VIDEO_FPS:-24.0}"
DEVICE="${DEVICE:-cuda:0}"
SEED_BASE="${SEED_BASE:-1000}"
RANDOM_SEED="${RANDOM_SEED:-1111}"

if [ -z "${STUDENT_CKPT_PATH}" ]; then
  echo "[run_mova_infer_T2A_random10] STUDENT_CKPT_PATH is empty — running with original MOVA weights only (no distillation)."
fi

if [ -z "${SELECTED_PROMPTS_FILE}" ] && [ ! -f "${PROMPT_FILE}" ]; then
  echo "[run_mova_infer_T2A_random10] Error: PROMPT_FILE not found: ${PROMPT_FILE}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

cd "${REPO_ROOT}"
export MOVA_ROOT
export PYTHONPATH="${REPO_ROOT}:${MOVA_ROOT}:${PYTHONPATH:-}"

if [ ! -x "${PYTHON_BIN}"; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "[run_mova_infer_T2A_random10] Error: no Python executable found."
    exit 1
  fi
fi
echo "[run_mova_infer_T2A_random10] python_bin=${PYTHON_BIN}"

if [ -n "${SELECTED_PROMPTS_FILE}" ]; then
  if [ ! -f "${SELECTED_PROMPTS_FILE}" ]; then
    echo "[run_mova_infer_T2A_random10] Error: SELECTED_PROMPTS_FILE not found: ${SELECTED_PROMPTS_FILE}"
    exit 1
  fi
  mapfile -t PROMPTS < <(awk 'NF {print}' "${SELECTED_PROMPTS_FILE}")
  echo "[run_mova_infer_T2A_random10] loaded ${#PROMPTS[@]} prompts from SELECTED_PROMPTS_FILE=${SELECTED_PROMPTS_FILE}"
else
  mapfile -t PROMPTS < <("${PYTHON_BIN}" - "${PROMPT_FILE}" "${NUM_SAMPLES}" "${RANDOM_SEED}" <<'PY'
import random
import sys

prompt_file = sys.argv[1]
num_samples = int(sys.argv[2])
seed = int(sys.argv[3])
with open(prompt_file, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]
if not prompts:
    sys.exit(0)
rng = random.Random(seed)
if num_samples >= len(prompts):
    selected = prompts
else:
    selected = rng.sample(prompts, num_samples)
for p in selected:
    print(p)
PY
)
  selected_path="${OUTPUT_DIR}/selected_prompts.txt"
  printf "%s\n" "${PROMPTS[@]}" > "${selected_path}"
  echo "[run_mova_infer_T2A_random10] sampled ${#PROMPTS[@]} prompts from ${PROMPT_FILE} with RANDOM_SEED=${RANDOM_SEED}"
  echo "[run_mova_infer_T2A_random10] saved selected prompts to ${selected_path}"
fi

if [ "${#PROMPTS[@]}" -eq 0 ]; then
  echo "[run_mova_infer_T2A_random10] Error: no valid prompts available."
  exit 1
fi

echo "[run_mova_infer_T2A_random10] output_dir=${OUTPUT_DIR}"
echo "[run_mova_infer_T2A_random10] steps=${NUM_INFERENCE_STEPS} cfg_scale=${CFG_SCALE} device=${DEVICE}"

batch_prompt_file="${OUTPUT_DIR}/batch_prompts.txt"
printf "%s\n" "${PROMPTS[@]}" > "${batch_prompt_file}"
echo "[run_mova_infer_T2A_random10] saved batch prompts to ${batch_prompt_file}"

STUDENT_ARGS=()
if [ -n "${STUDENT_CKPT_PATH}" ]; then
  STUDENT_ARGS=(--student_ckpt_path "${STUDENT_CKPT_PATH}")
fi

"${PYTHON_BIN}" "${PY_SCRIPT}" \
  --ckpt_path "${CKPT_PATH}" \
  "${STUDENT_ARGS[@]+"${STUDENT_ARGS[@]}"}" \
  --prompt_file "${batch_prompt_file}" \
  --output_dir "${OUTPUT_DIR}" \
  --save_prefix "sample_" \
  --write_prompt_txt \
  --seed_base "${SEED_BASE}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --cfg_scale "${CFG_SCALE}" \
  --num_frames "${NUM_FRAMES}" \
  --video_fps "${VIDEO_FPS}" \
  --device "${DEVICE}"

echo "[run_mova_infer_T2A_random10] done. results in ${OUTPUT_DIR}"
