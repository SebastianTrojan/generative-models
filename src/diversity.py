from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .generate import generate_batch, load_model
from .utils import (
    denormalize,
    get_device,
    load_yaml,
    output_root,
    resolve_path,
    save_json,
    save_tensor_grid,
    set_seed,
    timestamp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute simple diversity diagnostics for generated images.")
    parser.add_argument("--model", choices=["dcgan", "vae", "ddpm"], default="dcgan", help="Model type.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--num-images", type=int, default=256, help="Number of images to generate.")
    parser.add_argument("--batch-size", type=int, default=64, help="Generation batch size.")
    parser.add_argument("--max-pairwise-images", type=int, default=256, help="Limit for pairwise distance computation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda, cuda:0, or cpu.")
    return parser.parse_args()


def generate_images(model_name: str, model, config: dict, num_images: int, batch_size: int, device: torch.device) -> torch.Tensor:
    images: list[torch.Tensor] = []
    generated = 0
    with torch.no_grad():
        while generated < num_images:
            current_batch = min(batch_size, num_images - generated)
            batch = generate_batch(model_name, model, current_batch, config, device)
            images.append(batch.cpu())
            generated += current_batch
    return torch.cat(images, dim=0)


def diversity_metrics(images: torch.Tensor, max_pairwise_images: int) -> dict:
    images_01 = denormalize(images).cpu()
    subset = images_01[: min(max_pairwise_images, images_01.size(0))]
    flat = subset.view(subset.size(0), -1)
    pairwise = torch.pdist(flat, p=2)
    dimensions = flat.size(1)
    mean_l2 = pairwise.mean().item() if pairwise.numel() else 0.0
    std_l2 = pairwise.std(unbiased=False).item() if pairwise.numel() else 0.0
    rmse = mean_l2 / (dimensions**0.5)
    pixel_std = flat.std(dim=0, unbiased=False).mean().item()
    channel_std = images_01.flatten(2).std(dim=2, unbiased=False).mean(dim=0).tolist()
    warning = rmse < 0.05 or pixel_std < 0.03
    return {
        "num_images": int(images_01.size(0)),
        "pairwise_images_used": int(subset.size(0)),
        "average_pairwise_l2": mean_l2,
        "std_pairwise_l2": std_l2,
        "average_pairwise_rmse": rmse,
        "mean_pixel_std": pixel_std,
        "mean_channel_std": {
            "red": channel_std[0],
            "green": channel_std[1],
            "blue": channel_std[2],
        },
        "possible_mode_collapse_warning": warning,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    config = load_yaml(args.config)
    checkpoint_path = resolve_path(args.checkpoint, args.config, must_exist=True)
    model, model_config = load_model(args.model, checkpoint_path, args.config, device)
    model_config.update({key: value for key, value in config.items() if key not in model_config or key == "dataset_root"})

    out_dir = output_root(model_config) / "diversity" / model_config.get("run_name", args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = generate_images(args.model, model, model_config, args.num_images, args.batch_size, device)
    metrics = diversity_metrics(images, args.max_pairwise_images)
    metrics.update(
        {
            "model": args.model,
            "run_name": model_config.get("run_name", args.model),
            "checkpoint_path": str(checkpoint_path),
            "timestamp": timestamp(),
            "note": "Pixel-distance metrics are lightweight diagnostics; inspect grids and FID as well.",
        }
    )
    save_tensor_grid(images[: min(args.num_images, 64)], out_dir / "sample_grid.png")
    save_json(metrics, out_dir / "diversity.json")
    print(f"Saved diversity metrics to {out_dir / 'diversity.json'}")
    if metrics["possible_mode_collapse_warning"]:
        print("Warning: low diversity metrics may indicate mode collapse. Inspect the generated grid.")


if __name__ == "__main__":
    main()
