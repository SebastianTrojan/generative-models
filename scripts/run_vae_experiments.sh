#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS_OVERRIDE=""
TRAIN_ARGS=()

CONFIGS=(
  configs/vae_cat64_baseline.yaml
  configs/vae_cat64_prior.yaml
  configs/vae_cat64_beta025.yaml
  configs/vae_cat64_beta05.yaml
  configs/vae_cat64_latent256.yaml
  configs/vae_cat64_latent512.yaml
  configs/vae_cat64_refine2.yaml
  configs/vae_cat64_no_multiscale.yaml
)

usage() {
  printf '%s\n' "Usage: bash scripts/run_vae_experiments.sh [--epochs N] [train_vae args]"
  printf '%s\n' ""
  printf '%s\n' "Examples:"
  printf '%s\n' "  bash scripts/run_vae_experiments.sh"
  printf '%s\n' "  bash scripts/run_vae_experiments.sh --epochs 30"
  printf '%s\n' "  bash scripts/run_vae_experiments.sh --device cuda"
  printf '%s\n' ""
  printf '%s\n' "Set PYTHON_BIN to use a specific interpreter, for example:"
  printf '%s\n' "  PYTHON_BIN=./venv/bin/python bash scripts/run_vae_experiments.sh --epochs 30"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs)
      EPOCHS_OVERRIDE="${2:?Missing value for --epochs}"
      shift 2
      ;;
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

make_temp_config() {
  local config_path="$1"
  local epochs="$2"
  local tmp_dir="${TMPDIR:-/tmp}/vae_experiment_configs"
  local config_name
  config_name="$(basename "$config_path")"
  mkdir -p "$tmp_dir"
  local tmp_path="$tmp_dir/${config_name%.yaml}_e${epochs}.yaml"

  "$PYTHON_BIN" - "$config_path" "$tmp_path" "$epochs" <<'PY'
from pathlib import Path
import sys
import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
epochs = int(sys.argv[3])

with src.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

config["epochs"] = epochs
config["run_name"] = f"{config.get('run_name', src.stem)}_e{epochs}"

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)

print(dst)
PY
}

for config in "${CONFIGS[@]}"; do
  config_to_run="$config"
  if [[ -n "$EPOCHS_OVERRIDE" ]]; then
    config_to_run="$(make_temp_config "$config" "$EPOCHS_OVERRIDE")"
  fi

  printf '\n%s\n' "============================================================"
  printf '%s\n' "Training VAE config: $config_to_run"
  printf '%s\n' "============================================================"
  "$PYTHON_BIN" -m src.train_vae --config "$config_to_run" "${TRAIN_ARGS[@]}"
done
