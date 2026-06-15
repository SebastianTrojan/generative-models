#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
RESUME="${RESUME:-0}"
TRAIN_ARGS=()

usage() {
  printf '%s\n' "Usage: bash scripts/run_catdog_best_models.sh [train args]"
  printf '%s\n' ""
  printf '%s\n' "Examples:"
  printf '%s\n' "  PYTHON_BIN=./venv/bin/python bash scripts/run_catdog_best_models.sh --device cuda"
  printf '%s\n' "  RESUME=1 PYTHON_BIN=./venv/bin/python bash scripts/run_catdog_best_models.sh --device cuda"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      TRAIN_ARGS+=("$1")
      shift
      ;;
  esac
done

run_training() {
  local module="$1"
  local config="$2"
  local run_name="$3"
  local resume_path="outputs/checkpoints/${run_name}/training_latest.pt"
  local resume_args=()

  if [[ "$RESUME" == "1" && -f "$resume_path" ]]; then
    resume_args=(--resume "$resume_path")
  fi

  printf '\n%s\n' "============================================================"
  printf '%s\n' "Training ${module}: ${config}"
  if [[ "${#resume_args[@]}" -gt 0 ]]; then
    printf '%s\n' "Resuming from: ${resume_path}"
  fi
  printf '%s\n' "============================================================"

  "$PYTHON_BIN" -m "$module" --config "$config" "${resume_args[@]}" "${TRAIN_ARGS[@]}"
}

run_training src.train_dcgan configs/dcgan_catdog64_lr_d1e4.yaml dcgan_catdog64_lr_d1e4
run_training src.train_vae configs/vae_catdog64_beta05.yaml vae_catdog64_beta05
run_training src.train_ddpm configs/ddpm_catdog64_wide96.yaml ddpm_catdog64_wide96
