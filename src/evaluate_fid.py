from __future__ import annotations

import argparse
import hashlib
from datetime import datetime
from pathlib import Path

import torch

from .data import export_preprocessed_images
from .generate import generate_batch, load_model
from .utils import (
    get_device,
    load_yaml,
    output_root,
    resolve_path,
    save_image_batch,
    save_json,
    set_seed,
    timestamp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute FID for a trained generator checkpoint.")
    parser.add_argument("--model", choices=["dcgan", "vae", "ddpm"], required=True, help="Model type.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--config", required=True, help="Path to the YAML config used for dataset settings.")
    parser.add_argument("--num-images", type=int, default=5000, help="Number of generated images for FID.")
    parser.add_argument("--batch-size", type=int, default=64, help="Generation and FID batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Worker count for preprocessing/FID backend.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--sample-temperature",
        type=float,
        default=None,
        help="Override VAE latent sampling temperature for generated FID images.",
    )
    return parser.parse_args()


def safe_suffix() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_fid_images(model_name: str, model, config: dict, out_dir: Path, num_images: int, batch_size: int, device: torch.device) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    with torch.no_grad():
        while generated < num_images:
            current_batch = min(batch_size, num_images - generated)
            images = generate_batch(model_name, model, current_batch, config, device)
            save_image_batch(images, out_dir, prefix="fake", start_index=generated)
            generated += current_batch


def compute_fid_score(real_dir: Path, fake_dir: Path, batch_size: int, num_workers: int, device: torch.device) -> tuple[float, str]:
    try:
        from cleanfid import fid

        try:
            score = fid.compute_fid(
                str(real_dir),
                str(fake_dir),
                mode="clean",
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
            )
        except TypeError:
            score = fid.compute_fid(str(real_dir), str(fake_dir), mode="clean")
        return float(score), "clean-fid"
    except ImportError:
        pass

    try:
        from pytorch_fid import fid_score

        paths = [str(real_dir), str(fake_dir)]
        try:
            score = fid_score.calculate_fid_given_paths(
                paths,
                batch_size=batch_size,
                device=device,
                dims=2048,
                num_workers=num_workers,
            )
        except TypeError:
            score = fid_score.calculate_fid_given_paths(paths, batch_size, device, 2048)
        return float(score), "pytorch-fid"
    except ImportError as exc:
        raise RuntimeError(
            "FID backend is unavailable. Install clean-fid or pytorch-fid, e.g. "
            "`pip install clean-fid`."
        ) from exc


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    config = load_yaml(args.config)
    checkpoint_path = resolve_path(args.checkpoint, args.config, must_exist=True)
    model, model_config = load_model(args.model, checkpoint_path, args.config, device)
    model_config.update({key: value for key, value in config.items() if key not in model_config or key == "dataset_root"})
    if args.model == "vae" and args.sample_temperature is not None:
        model_config["sample_temperature"] = args.sample_temperature

    dataset_root = resolve_path(config["dataset_root"], args.config, must_exist=True)
    image_size = int(model_config.get("image_size", config.get("image_size", 64)))
    fid_root = output_root(model_config) / "fid"
    dataset_hash = hashlib.sha1(str(dataset_root).encode("utf-8")).hexdigest()[:8]
    real_dir = fid_root / f"real_{dataset_hash}_{image_size}_{args.num_images}"
    fake_dir = fid_root / f"{model_config.get('run_name', args.model)}_{args.model}_fake_{args.num_images}_{safe_suffix()}"

    real_count = export_preprocessed_images(dataset_root, real_dir, image_size=image_size, limit=args.num_images)
    if real_count < args.num_images:
        raise RuntimeError(f"Only {real_count} valid real images were exported; requested {args.num_images}.")
    generate_fid_images(args.model, model, model_config, fake_dir, args.num_images, args.batch_size, device)

    fid_value, backend = compute_fid_score(real_dir, fake_dir, args.batch_size, args.num_workers, device)
    result = {
        "model": args.model,
        "run_name": model_config.get("run_name", args.model),
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(dataset_root),
        "num_generated_images": args.num_images,
        "image_size": image_size,
        "fid": fid_value,
        "backend": backend,
        "sample_temperature": model_config.get("sample_temperature") if args.model == "vae" else None,
        "real_dir": str(real_dir),
        "fake_dir": str(fake_dir),
        "timestamp": timestamp(),
    }
    out_path = fid_root / f"{model_config.get('run_name', args.model)}_fid.json"
    save_json(result, out_path)
    print(f"FID ({backend}) = {fid_value:.4f}")
    print(f"Saved FID JSON to {out_path}")


if __name__ == "__main__":
    main()
