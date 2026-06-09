from __future__ import annotations

import csv
import json
import random
import shutil
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from torchvision.utils import make_grid, save_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    data["_config_path"] = str(path.resolve())
    return data


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    cleaned = {k: v for k, v in data.items() if not k.startswith("_")}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cleaned, handle, sort_keys=False)


def resolve_path(path: str | Path, config_path: str | Path | None = None, must_exist: bool = False) -> Path:
    """Resolve user paths from common launch locations.

    Relative paths are checked against the config directory, this project root,
    the workspace root, and the current working directory. If none exist, the
    path is resolved under the project root.
    """

    raw = Path(path).expanduser()
    if raw.is_absolute():
        resolved = raw
    else:
        bases: list[Path] = []
        if config_path:
            bases.append(Path(config_path).expanduser().resolve().parent)
        bases.extend([PROJECT_ROOT, PROJECT_ROOT.parent, Path.cwd()])
        for base in bases:
            candidate = (base / raw).resolve()
            if candidate.exists():
                return candidate
        resolved = (PROJECT_ROOT / raw).resolve()

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved


def output_root(config: dict[str, Any]) -> Path:
    return resolve_path(config.get("output_dir", "outputs"), config.get("_config_path"))


def run_dir(config: dict[str, Any], kind: str) -> Path:
    path = output_root(config) / kind / config["run_name"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_dir(config: dict[str, Any]) -> Path:
    return run_dir(config, "checkpoints")


def sample_dir(config: dict[str, Any]) -> Path:
    return run_dir(config, "samples")


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def use_mixed_precision(config: dict[str, Any], device: torch.device) -> bool:
    return bool(config.get("mixed_precision", False)) and device.type == "cuda"


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def make_grad_scaler(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device.type, enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def denormalize(images: torch.Tensor) -> torch.Tensor:
    return images.detach().mul(0.5).add(0.5).clamp(0.0, 1.0)


def save_tensor_grid(
    images: torch.Tensor,
    path: str | Path,
    nrow: int | None = None,
    normalize_from_tanh: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = denormalize(images) if normalize_from_tanh else images.detach().clamp(0.0, 1.0)
    if nrow is None:
        nrow = max(1, int(np.sqrt(images.size(0))))
    grid = make_grid(images.cpu(), nrow=nrow, padding=2)
    save_image(grid, path)


def save_image_batch(
    images: torch.Tensor,
    out_dir: str | Path,
    prefix: str = "sample",
    start_index: int = 0,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = denormalize(images).cpu()
    for index, image in enumerate(images, start=start_index):
        save_image(image, out_dir / f"{prefix}_{index:05d}.png")


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_metrics_csv(path: str | Path, fieldnames: Iterable[str], row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def copy_config(config: dict[str, Any], destination: str | Path) -> None:
    config_path = config.get("_config_path")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if config_path and Path(config_path).exists():
        shutil.copy2(config_path, destination)
    else:
        save_yaml(config, destination)


def clean_config(config: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if not k.startswith("_")}


def torch_load(path: str | Path, device: torch.device | str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def checkpoint_config(checkpoint: dict[str, Any], fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    config = dict(fallback or {})
    config.update(checkpoint.get("config", {}))
    return config
