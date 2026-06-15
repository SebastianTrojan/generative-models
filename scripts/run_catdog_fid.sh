#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DCGAN_NUM_IMAGES="${DCGAN_NUM_IMAGES:-5000}"
VAE_NUM_IMAGES="${VAE_NUM_IMAGES:-5000}"
DDPM_NUM_IMAGES="${DDPM_NUM_IMAGES:-2500}"
DCGAN_BATCH_SIZE="${DCGAN_BATCH_SIZE:-64}"
VAE_BATCH_SIZE="${VAE_BATCH_SIZE:-64}"
DDPM_BATCH_SIZE="${DDPM_BATCH_SIZE:-8}"
VAE_TEMP="${VAE_TEMP:-0.65}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SUMMARY_PATH="${FID_SUMMARY:-outputs/fid/fid_catdog_summary_$(date +%Y%m%d_%H%M%S).csv}"
EVAL_ARGS=()

usage() {
  printf '%s\n' "Usage: bash scripts/run_catdog_fid.sh [evaluate_fid args]"
  printf '%s\n' ""
  printf '%s\n' "Defaults:"
  printf '%s\n' "  DCGAN: ${DCGAN_NUM_IMAGES} images, batch ${DCGAN_BATCH_SIZE}"
  printf '%s\n' "  VAE:   ${VAE_NUM_IMAGES} images, batch ${VAE_BATCH_SIZE}, temperature ${VAE_TEMP}"
  printf '%s\n' "  DDPM:  ${DDPM_NUM_IMAGES} images, batch ${DDPM_BATCH_SIZE}"
  printf '%s\n' ""
  printf '%s\n' "Examples:"
  printf '%s\n' "  PYTHON_BIN=./venv/bin/python bash scripts/run_catdog_fid.sh --device cuda"
  printf '%s\n' "  VAE_TEMP=0.80 PYTHON_BIN=./venv/bin/python bash scripts/run_catdog_fid.sh --device cuda"
  printf '%s\n' "  SKIP_EXISTING=0 PYTHON_BIN=./venv/bin/python bash scripts/run_catdog_fid.sh --device cuda"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EVAL_ARGS+=("$1")
      shift
      ;;
  esac
done

mkdir -p outputs/fid
printf '%s\n' "method,run_name,config,checkpoint,num_images,sample_temperature,fid,backend,result_json" > "$SUMMARY_PATH"

sanitize_temp() {
  local temp="$1"
  temp="${temp//./p}"
  temp="${temp//-/m}"
  printf '%s' "$temp"
}

record_result() {
  local method="$1"
  local config="$2"
  local checkpoint="$3"
  local result_json="$4"

  "$PYTHON_BIN" - "$method" "$config" "$checkpoint" "$result_json" "$SUMMARY_PATH" <<'PY'
import csv
import json
import sys
from pathlib import Path

method, config_path, checkpoint_path, result_path, summary_path = sys.argv[1:]
result = json.loads(Path(result_path).read_text())

row = {
    "method": method,
    "run_name": result.get("run_name", ""),
    "config": config_path,
    "checkpoint": checkpoint_path,
    "num_images": result.get("num_generated_images", ""),
    "sample_temperature": "" if result.get("sample_temperature") is None else result.get("sample_temperature"),
    "fid": result.get("fid", ""),
    "backend": result.get("backend", ""),
    "result_json": result_path,
}

with Path(summary_path).open("a", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(row))
    writer.writerow(row)
PY
}

run_fid() {
  local method="$1"
  local run_name="$2"
  local config="$3"
  local checkpoint="$4"
  local num_images="$5"
  local batch_size="$6"
  local result_json="$7"
  shift 7
  local extra_args=("$@")

  if [[ ! -f "$checkpoint" ]]; then
    printf '%s\n' "Skipping ${run_name}: checkpoint not found: ${checkpoint}"
    return
  fi

  if [[ "$SKIP_EXISTING" == "1" && -f "$result_json" ]]; then
    printf '%s\n' "Skipping ${run_name}: existing result ${result_json}"
    record_result "$method" "$config" "$checkpoint" "$result_json"
    return
  fi

  printf '\n%s\n' "============================================================"
  printf '%s\n' "Cat+dog FID ${method}: ${run_name}"
  printf '%s\n' "Config: ${config}"
  printf '%s\n' "Checkpoint: ${checkpoint}"
  printf '%s\n' "Images: ${num_images}; batch: ${batch_size}"
  if [[ "${#extra_args[@]}" -gt 0 ]]; then
    printf '%s\n' "Extra args: ${extra_args[*]}"
  fi
  printf '%s\n' "============================================================"

  "$PYTHON_BIN" -m src.evaluate_fid \
    --model "$method" \
    --checkpoint "$checkpoint" \
    --config "$config" \
    --num-images "$num_images" \
    --batch-size "$batch_size" \
    --num-workers "$NUM_WORKERS" \
    "${extra_args[@]}" \
    "${EVAL_ARGS[@]}"

  local default_json="outputs/fid/${run_name}_fid.json"
  if [[ ! -f "$default_json" ]]; then
    printf '%s\n' "Expected FID result not found: ${default_json}" >&2
    exit 1
  fi

  cp "$default_json" "$result_json"
  record_result "$method" "$config" "$checkpoint" "$result_json"
}

temp_safe="$(sanitize_temp "$VAE_TEMP")"

run_fid \
  dcgan \
  dcgan_catdog64_lr_d1e4 \
  configs/dcgan_catdog64_lr_d1e4.yaml \
  outputs/checkpoints/dcgan_catdog64_lr_d1e4/generator_latest.pt \
  "$DCGAN_NUM_IMAGES" \
  "$DCGAN_BATCH_SIZE" \
  "outputs/fid/dcgan_catdog64_lr_d1e4_latest_n${DCGAN_NUM_IMAGES}_fid.json"

run_fid \
  vae \
  vae_catdog64_beta05 \
  configs/vae_catdog64_beta05.yaml \
  outputs/checkpoints/vae_catdog64_beta05/model_latest.pt \
  "$VAE_NUM_IMAGES" \
  "$VAE_BATCH_SIZE" \
  "outputs/fid/vae_catdog64_beta05_latest_t${temp_safe}_n${VAE_NUM_IMAGES}_fid.json" \
  --sample-temperature "$VAE_TEMP"

run_fid \
  ddpm \
  ddpm_catdog64_wide96 \
  configs/ddpm_catdog64_wide96.yaml \
  outputs/checkpoints/ddpm_catdog64_wide96/model_latest.pt \
  "$DDPM_NUM_IMAGES" \
  "$DDPM_BATCH_SIZE" \
  "outputs/fid/ddpm_catdog64_wide96_latest_n${DDPM_NUM_IMAGES}_fid.json"

printf '\n%s\n' "Cat+dog FID sweep complete."
printf '%s\n' "Summary: ${SUMMARY_PATH}"
