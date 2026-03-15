#!/bin/bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_mova_infer_T2A.sh [options]

Options:
  --prompt "<text>"         Single prompt text mode
  --prompt-file <path>      Single prompt file mode (one txt file, multiple lines allowed)
  --output-dir <path>       Output directory for batch mode
  --save-path <path>        Output wav path for single prompt mode
  -h, --help                Show this help

Environment fallback:
  PROMPT="<text>"
  PROMPT_FILE="/path/one.txt"
  DEFAULT_PROMPT_TEXT="<default prompt content when no prompt args are provided>"

Notes:
  1) Two modes only: PROMPT or PROMPT_FILE.
  2) No random sampling.
  3) In PROMPT mode, default save path is OUTPUT_DIR/output.wav when SAVE_PATH is not set.
  4) In PROMPT_FILE mode, outputs are written to OUTPUT_DIR.
  5) If no prompt args are provided, script auto-creates OUTPUT_DIR/default_prompt.txt and runs it.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PY_SCRIPT="${REPO_ROOT}/turbodiffusion/inference/mova_t2a_infer.py"
MOVA_ROOT="${MOVA_ROOT:-/home/jovyan/codes/turbodiff/MOVA}"
PYTHON_BIN="${PYTHON_BIN:-/home/jovyan/codes/turbodiff/TurboDiff-T2AV/.pixi/envs/default/bin/python}"

CKPT_PATH="${CKPT_PATH:-/data/datasets/turbodiff_datasets_and_ckpt/MOVA_checkpoints/MOVA-360p}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/t2a_output}"
SAVE_PATH="${SAVE_PATH:-}"
SAVE_PREFIX="${SAVE_PREFIX:-sample_}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
CFG_SCALE="${CFG_SCALE:-5.0}"
NUM_FRAMES="${NUM_FRAMES:-193}"
VIDEO_FPS="${VIDEO_FPS:-24.0}"
DEVICE="${DEVICE:-cuda:0}"
SEED_BASE="${SEED_BASE:-1000}"
SEED="${SEED:-42}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-noise, distorted, low quality, bad audio, muffled, static}"
WRITE_PROMPT_TXT="${WRITE_PROMPT_TXT:-1}"
DEFAULT_PROMPT_TEXT="${DEFAULT_PROMPT_TEXT:-The music piece is instrumental with a prominent keyboard sound, blending elements of electronic and classical genres. It creates an inspiring mood, utilizing a 4/4 time signature and a tempo of approximately 80 bpm, suitable for scenarios depicting determination or overcoming obstacles.}"

INPUT_PROMPT=""
INPUT_PROMPT_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt)
      if [[ $# -lt 2 ]]; then
        echo "[run_mova_infer_T2A] Error: --prompt requires a value."
        exit 1
      fi
      if [[ -n "${INPUT_PROMPT}" ]]; then
        echo "[run_mova_infer_T2A] Error: --prompt can only be provided once."
        exit 1
      fi
      INPUT_PROMPT="$2"
      shift 2
      ;;
    --prompt-file)
      if [[ $# -lt 2 ]]; then
        echo "[run_mova_infer_T2A] Error: --prompt-file requires a value."
        exit 1
      fi
      if [[ -n "${INPUT_PROMPT_FILE}" ]]; then
        echo "[run_mova_infer_T2A] Error: --prompt-file can only be provided once."
        exit 1
      fi
      INPUT_PROMPT_FILE="$2"
      shift 2
      ;;
    --output-dir)
      if [[ $# -lt 2 ]]; then
        echo "[run_mova_infer_T2A] Error: --output-dir requires a value."
        exit 1
      fi
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --save-path)
      if [[ $# -lt 2 ]]; then
        echo "[run_mova_infer_T2A] Error: --save-path requires a value."
        exit 1
      fi
      SAVE_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[run_mova_infer_T2A] Error: unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

# Environment fallback.
if [[ -z "${INPUT_PROMPT}" && -n "${PROMPT:-}" ]]; then
  INPUT_PROMPT="${PROMPT}"
fi
if [[ -z "${INPUT_PROMPT_FILE}" && -n "${PROMPT_FILE:-}" ]]; then
  INPUT_PROMPT_FILE="${PROMPT_FILE}"
fi

if [[ -z "${INPUT_PROMPT}" && -z "${INPUT_PROMPT_FILE}" ]]; then
  mkdir -p "${OUTPUT_DIR}"
  INPUT_PROMPT_FILE="${OUTPUT_DIR}/default_prompt.txt"
  printf "%s\n" "${DEFAULT_PROMPT_TEXT}" > "${INPUT_PROMPT_FILE}"
  echo "[run_mova_infer_T2A] No prompt args provided, using default prompt."
fi

if [[ -n "${INPUT_PROMPT}" && -n "${INPUT_PROMPT_FILE}" ]]; then
  echo "[run_mova_infer_T2A] Error: --prompt and --prompt-file are mutually exclusive."
  usage
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

cd "${REPO_ROOT}"
export MOVA_ROOT
export PYTHONPATH="${REPO_ROOT}:${MOVA_ROOT}:${PYTHONPATH:-}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "[run_mova_infer_T2A] Error: no Python executable found."
    exit 1
  fi
fi

echo "[run_mova_infer_T2A] python_bin=${PYTHON_BIN}"
echo "[run_mova_infer_T2A] output_dir=${OUTPUT_DIR}"
echo "[run_mova_infer_T2A] steps=${NUM_INFERENCE_STEPS} cfg_scale=${CFG_SCALE} device=${DEVICE}"

WRITE_PROMPT_ARGS=()
if [[ "${WRITE_PROMPT_TXT}" == "1" || "${WRITE_PROMPT_TXT}" == "true" || "${WRITE_PROMPT_TXT}" == "TRUE" ]]; then
  WRITE_PROMPT_ARGS=(--write_prompt_txt)
fi

if [[ -n "${INPUT_PROMPT}" ]]; then
  if [[ ! "${INPUT_PROMPT}" =~ [^[:space:]] ]]; then
    echo "[run_mova_infer_T2A] Error: --prompt is empty."
    exit 1
  fi
  if [[ -z "${SAVE_PATH}" ]]; then
    SAVE_PATH="${OUTPUT_DIR}/output.wav"
    echo "[run_mova_infer_T2A] SAVE_PATH not set in single prompt mode, using default: ${SAVE_PATH}"
  fi

  single_prompt="${INPUT_PROMPT}"
  mkdir -p "$(dirname "${SAVE_PATH}")"
  "${PYTHON_BIN}" "${PY_SCRIPT}" \
    --ckpt_path "${CKPT_PATH}" \
    --prompt "${single_prompt}" \
    --save_path "${SAVE_PATH}" \
    --negative_prompt "${NEGATIVE_PROMPT}" \
    --seed "${SEED}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --cfg_scale "${CFG_SCALE}" \
    --num_frames "${NUM_FRAMES}" \
    --video_fps "${VIDEO_FPS}" \
    --device "${DEVICE}"

  if [[ "${#WRITE_PROMPT_ARGS[@]}" -gt 0 ]]; then
    prompt_txt_path="${SAVE_PATH%.*}.txt"
    printf "%s\n" "${single_prompt}" > "${prompt_txt_path}"
  fi
  echo "[run_mova_infer_T2A] done(single). result=${SAVE_PATH}"
else
  if [[ ! -f "${INPUT_PROMPT_FILE}" ]]; then
    echo "[run_mova_infer_T2A] Error: prompt file not found: ${INPUT_PROMPT_FILE}"
    exit 1
  fi
  if [[ -n "${SAVE_PATH}" ]]; then
    echo "[run_mova_infer_T2A] Warning: SAVE_PATH is ignored in prompt-file mode."
  fi

  batch_prompt_file="${OUTPUT_DIR}/batch_prompts.txt"
  mapfile -t _loaded_prompts < <(awk 'NF {print}' "${INPUT_PROMPT_FILE}")
  if [[ "${#_loaded_prompts[@]}" -eq 0 ]]; then
    echo "[run_mova_infer_T2A] Error: no valid prompts in ${INPUT_PROMPT_FILE}"
    exit 1
  fi
  printf "%s\n" "${_loaded_prompts[@]}" > "${batch_prompt_file}"
  echo "[run_mova_infer_T2A] saved batch prompts to ${batch_prompt_file}"

  "${PYTHON_BIN}" "${PY_SCRIPT}" \
    --ckpt_path "${CKPT_PATH}" \
    --prompt_file "${batch_prompt_file}" \
    --output_dir "${OUTPUT_DIR}" \
    --save_prefix "${SAVE_PREFIX}" \
    "${WRITE_PROMPT_ARGS[@]+"${WRITE_PROMPT_ARGS[@]}"}" \
    --seed_base "${SEED_BASE}" \
    --negative_prompt "${NEGATIVE_PROMPT}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --cfg_scale "${CFG_SCALE}" \
    --num_frames "${NUM_FRAMES}" \
    --video_fps "${VIDEO_FPS}" \
    --device "${DEVICE}"

  echo "[run_mova_infer_T2A] done(batch). results in ${OUTPUT_DIR}"
fi
