from __future__ import annotations

import argparse

import numpy as np
import torch

from .generate import load_model
from .utils import (
    get_device,
    output_root,
    resolve_path,
    save_image_batch,
    save_tensor_grid,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interpolate between two latent/noise tensors.")
    parser.add_argument("--model", choices=["dcgan", "vae", "ddpm"], default="dcgan", help="Model type.")
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint.")
    parser.add_argument("--config", default=None, help="Optional config path.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to outputs/interpolations/<run_name>.")
    parser.add_argument("--steps", type=int, default=10, help="Total interpolation images, including endpoints.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--sample-temperature",
        type=float,
        default=None,
        help="Override VAE latent sampling temperature for interpolation endpoints.",
    )
    return parser.parse_args()


def interpolate_tensors(a: torch.Tensor, b: torch.Tensor, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    coefficient_shape = (steps,) + (1,) * (a.dim() - 1)
    coefficients = torch.linspace(0.0, 1.0, steps, device=a.device).view(coefficient_shape)
    interpolated = (1.0 - coefficients) * a + coefficients * b
    return interpolated, coefficients.view(-1)


@torch.no_grad()
def generate_interpolation(model_name: str, model, config: dict, steps: int, device: torch.device):
    if model_name == "dcgan":
        latent_dim = int(config.get("latent_dim", 100))
        z_a = torch.randn(1, latent_dim, 1, 1, device=device)
        z_b = torch.randn(1, latent_dim, 1, 1, device=device)
        latents, coefficients = interpolate_tensors(z_a, z_b, steps)
        images = model(latents)
        return images, z_a, z_b, latents, coefficients

    if model_name == "vae":
        latent_dim = int(config.get("latent_dim", 128))
        temperature = float(config.get("sample_temperature", 1.0))
        z_a = torch.randn(1, latent_dim, device=device) * temperature
        z_b = torch.randn(1, latent_dim, device=device) * temperature
        latents, coefficients = interpolate_tensors(z_a, z_b, steps)
        images = model.decode(latents)
        return images, z_a, z_b, latents, coefficients

    channels = int(config.get("channels", 3))
    image_size = int(config.get("image_size", 64))
    z_a = torch.randn(1, channels, image_size, image_size, device=device)
    z_b = torch.randn(1, channels, image_size, image_size, device=device)
    latents, coefficients = interpolate_tensors(z_a, z_b, steps)
    images = model["diffusion"].ddim_sample_loop(
        model["denoiser"],
        shape=(steps, channels, image_size, image_size),
        device=device,
        initial_noise=latents,
        progress=True,
    )
    return images, z_a, z_b, latents, coefficients


def main() -> None:
    args = parse_args()
    if args.steps < 2:
        raise ValueError("--steps must be at least 2 so both endpoints are present.")
    set_seed(args.seed)
    device = get_device(args.device)
    checkpoint_path = resolve_path(args.checkpoint, args.config, must_exist=True)
    model, config = load_model(args.model, checkpoint_path, args.config, device)
    if args.model == "vae" and args.sample_temperature is not None:
        config["sample_temperature"] = args.sample_temperature

    if args.out_dir:
        out_dir = resolve_path(args.out_dir, config.get("_config_path"))
    else:
        out_dir = output_root(config) / "interpolations" / config.get("run_name", args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    images, z_a, z_b, latents, coefficients = generate_interpolation(args.model, model, config, args.steps, device)
    save_tensor_grid(images, out_dir / "interpolation_grid.png", nrow=args.steps)
    save_image_batch(images, out_dir, prefix="interp")
    np.savez(
        out_dir / "latent_vectors.npz",
        model=args.model,
        z_a=z_a.detach().cpu().numpy(),
        z_b=z_b.detach().cpu().numpy(),
        latents=latents.detach().cpu().numpy(),
        t=coefficients.detach().cpu().numpy(),
    )
    print(f"Saved {args.model} interpolation grid and latent/noise tensors to {out_dir}")


if __name__ == "__main__":
    main()
