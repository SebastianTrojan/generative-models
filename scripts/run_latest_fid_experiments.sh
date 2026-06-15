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
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SUMMARY_PATH="${FID_SUMMARY:-outputs/fid/fid_latest_experiments_summary_$(date +%Y%m%d_%H%M%S).csv}"
EVAL_ARGS=()

read -r -a VAE_TEMPS <<< "${VAE_TEMPS:-0.50 0.65 0.80}"

DCGAN_CONFIGS=(
  configs/dcgan_cat64_baseline.yaml
  configs/dcgan_cat64_lr_g1e4.yaml
  configs/dcgan_cat64_lr_d1e4.yaml
  configs/dcgan_cat64_z128.yaml
  configs/dcgan_cat64_g96.yaml
  configs/dcgan_cat64_no_noise.yaml
  configs/dcgan_cat64_smooth1.yaml
  configs/dcgan_cat64_mbstd.yaml
)

VAE_CONFIGS=(
  configs/vae_cat64_baseline.yaml
  configs/vae_cat64_prior.yaml
  configs/vae_cat64_beta025.yaml
  configs/vae_cat64_beta05.yaml
  configs/vae_cat64_latent256.yaml
  configs/vae_cat64_latent512.yaml
  configs/vae_cat64_refine2.yaml
  configs/vae_cat64_no_multiscale.yaml
)

DDPM_CONFIGS=(
  configs/ddpm_cat64_baseline.yaml
  configs/ddpm_cat64_cosine.yaml
  configs/ddpm_cat64_wide96.yaml
  configs/ddpm_cat64_1000steps.yaml
  configs/ddpm_cat64_lr1e4.yaml
  configs/ddpm_cat64_nodropout.yaml
)

usage() {
  printf '%s\n' "Usage: bash scripts/run_latest_fid_experiments.sh [evaluate_fid args]"
  printf '%s\n' ""
  printf '%s\n' "Defaults:"
  printf '%s\n' "  DCGAN: ${DCGAN_NUM_IMAGES} images, batch ${DCGAN_BATCH_SIZE}"
  printf '%s\n' "  VAE:   ${VAE_NUM_IMAGES} images, batch ${VAE_BATCH_SIZE}, temperatures: ${VAE_TEMPS[*]}"
  printf '%s\n' "  DDPM:  ${DDPM_NUM_IMAGES} images, batch ${DDPM_BATCH_SIZE}"
  printf '%s\n' ""
  printf '%s\n' "Examples:"
  printf '%s\n' "  PYTHON_BIN=./venv/bin/python bash scripts/run_latest_fid_experiments.sh --device cuda"
  printf '%s\n' "  VAE_TEMPS=\"0.50 0.65 0.80 1.00\" PYTHON_BIN=./venv/bin/python bash scripts/run_latest_fid_experiments.sh --device cuda"
  printf '%s\n' "  SKIP_EXISTING=0 PYTHON_BIN=./venv/bin/python bash scripts/run_latest_fid_experiments.sh --device cuda"
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

run_name_for_config() {
  "$PYTHON_BIN" - "$1" <<'PY'
from pathlib import Path
import sys
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
print(config["run_name"])
PY
}

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
  local config="$2"
  local checkpoint="$3"
  local num_images="$4"
  local batch_size="$5"
  local result_json="$6"
  shift 6
  local extra_args=("$@")
  local run_name
  run_name="$(run_name_for_config "$config")"

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
  printf '%s\n' "FID ${method}: ${run_name}"
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

for config in "${DCGAN_CONFIGS[@]}"; do
  run_name="$(run_name_for_config "$config")"
  run_fid \
    dcgan \
    "$config" \
    "outputs/checkpoints/${run_name}/generator_latest.pt" \
    "$DCGAN_NUM_IMAGES" \
    "$DCGAN_BATCH_SIZE" \
    "outputs/fid/${run_name}_latest_n${DCGAN_NUM_IMAGES}_fid.json"
done

for config in "${VAE_CONFIGS[@]}"; do
  run_name="$(run_name_for_config "$config")"
  for temp in "${VAE_TEMPS[@]}"; do
    temp_safe="$(sanitize_temp "$temp")"
    run_fid \
      vae \
      "$config" \
      "outputs/checkpoints/${run_name}/model_latest.pt" \
      "$VAE_NUM_IMAGES" \
      "$VAE_BATCH_SIZE" \
      "outputs/fid/${run_name}_latest_t${temp_safe}_n${VAE_NUM_IMAGES}_fid.json" \
      --sample-temperature "$temp"
  done
done

for config in "${DDPM_CONFIGS[@]}"; do
  run_name="$(run_name_for_config "$config")"
  run_fid \
    ddpm \
    "$config" \
    "outputs/checkpoints/${run_name}/model_latest.pt" \
    "$DDPM_NUM_IMAGES" \
    "$DDPM_BATCH_SIZE" \
    "outputs/fid/${run_name}_latest_n${DDPM_NUM_IMAGES}_fid.json"
done

printf '\n%s\n' "FID sweep complete."
printf '%s\n' "Summary: ${SUMMARY_PATH}"
