from __future__ import annotations

import argparse

import torch

from .data import build_dataloader
from .generate import load_model
from .utils import (
    get_device,
    resolve_path,
    save_image_batch,
    save_tensor_grid,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create VAE reconstruction grids from real images.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained VAE model checkpoint.")
    parser.add_argument("--config", required=True, help="Path to the VAE YAML config.")
    parser.add_argument("--num-images", type=int, default=64, help="Number of real images to reconstruct.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for reconstruction.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument("--out-dir", required=True, help="Directory for reconstruction PNGs and grids.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda, cuda:0, or cpu.")
    return parser.parse_args()


@torch.no_grad()
def reconstruct_images(model, loader, num_images: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    originals: list[torch.Tensor] = []
    reconstructions: list[torch.Tensor] = []
    collected = 0

    for batch in loader:
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch
        current = min(images.size(0), num_images - collected)
        if current <= 0:
            break

        images = images[:current].to(device, non_blocking=True)
        reconstruction, _, _ = model(images)
        originals.append(images.cpu())
        reconstructions.append(reconstruction.cpu())
        collected += current

        if collected >= num_images:
            break

    if not originals:
        raise RuntimeError("No images were reconstructed.")

    return torch.cat(originals, dim=0), torch.cat(reconstructions, dim=0)


def main() -> None:
    args = parse_args()
    if args.num_images < 1:
        raise ValueError("--num-images must be at least 1.")

    set_seed(args.seed)
    device = get_device(args.device)
    checkpoint_path = resolve_path(args.checkpoint, args.config, must_exist=True)
    model, config = load_model("vae", checkpoint_path, args.config, device)
    config["batch_size"] = args.batch_size
    config["num_workers"] = args.num_workers

    _, loader = build_dataloader(config, train=False)
    originals, reconstructions = reconstruct_images(model, loader, args.num_images, device)

    out_dir = resolve_path(args.out_dir, config.get("_config_path"))
    out_dir.mkdir(parents=True, exist_ok=True)
    nrow = max(1, int(args.num_images**0.5))

    paired = torch.stack((originals, reconstructions), dim=1).flatten(0, 1)
    save_tensor_grid(originals, out_dir / "originals_grid.png", nrow=nrow)
    save_tensor_grid(reconstructions, out_dir / "reconstructions_grid.png", nrow=nrow)
    save_tensor_grid(torch.cat([originals, reconstructions], dim=0), out_dir / "reconstruction_grid.png", nrow=nrow)
    save_tensor_grid(paired, out_dir / "paired_grid.png", nrow=nrow * 2)
    save_image_batch(originals, out_dir, prefix="real")
    save_image_batch(reconstructions, out_dir, prefix="recon")

    print(f"Saved {originals.size(0)} VAE reconstructions to {out_dir}")
    print(f"Main grid: {out_dir / 'reconstruction_grid.png'}")
    print(f"Paired grid: {out_dir / 'paired_grid.png'}")


if __name__ == "__main__":
    main()
