from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .models.dcgan import build_generator_from_config
from .models.ddpm import build_denoiser_from_config, build_diffusion_from_config
from .models.vae import build_vae_from_config
from .utils import (
    checkpoint_config,
    get_device,
    load_yaml,
    resolve_path,
    save_image_batch,
    save_tensor_grid,
    set_seed,
    torch_load,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images from a trained model checkpoint.")
    parser.add_argument("--model", choices=["dcgan", "vae", "ddpm"], required=True, help="Model type to load.")
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint.")
    parser.add_argument("--config", default=None, help="Optional config override.")
    parser.add_argument("--num-images", type=int, default=64, help="Number of images to generate.")
    parser.add_argument("--batch-size", type=int, default=None, help="Generation batch size.")
    parser.add_argument("--out-dir", required=True, help="Directory for PNG images and grid.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--sample-temperature",
        type=float,
        default=None,
        help="Override VAE latent sampling temperature. Lower values can improve coherence at the cost of diversity.",
    )
    return parser.parse_args()


def load_generation_config(checkpoint: dict, config_path: str | None) -> dict:
    fallback = load_yaml(config_path) if config_path else {}
    return checkpoint_config(checkpoint, fallback)


def load_model(model_name: str, checkpoint_path: str | Path, config_path: str | None, device: torch.device):
    checkpoint = torch_load(checkpoint_path, device)
    config = load_generation_config(checkpoint, config_path)
    if model_name == "dcgan":
        model = build_generator_from_config(config).to(device)
        state_dict = checkpoint.get("state_dict", checkpoint.get("generator_state_dict"))
    elif model_name == "ddpm":
        denoiser = build_denoiser_from_config(config).to(device)
        diffusion = build_diffusion_from_config(config, device=device)
        state_dict = checkpoint.get("state_dict")
        model = {"denoiser": denoiser, "diffusion": diffusion}
    else:
        model = build_vae_from_config(config).to(device)
        state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise ValueError(f"Checkpoint does not contain a compatible state_dict: {checkpoint_path}")
    if model_name == "ddpm":
        model["denoiser"].load_state_dict(state_dict)
        model["denoiser"].eval()
    else:
        model.load_state_dict(state_dict)
        model.eval()
    return model, config


def generate_batch(model_name: str, model, batch_size: int, config: dict, device: torch.device) -> torch.Tensor:
    if model_name == "dcgan":
        latent_dim = int(config.get("latent_dim", 100))
        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        return model(noise)
    if model_name == "ddpm":
        shape = (
            batch_size,
            int(config.get("channels", 3)),
            int(config.get("image_size", 64)),
            int(config.get("image_size", 64)),
        )
        return model["diffusion"].p_sample_loop(model["denoiser"], shape, device=device, progress=False)
    latent_dim = int(config.get("latent_dim", 128))
    temperature = float(config.get("sample_temperature", 1.0))
    z = torch.randn(batch_size, latent_dim, device=device) * temperature
    return model.decode(z)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    checkpoint_path = resolve_path(args.checkpoint, args.config, must_exist=True)
    model, config = load_model(args.model, checkpoint_path, args.config, device)
    if args.model == "vae" and args.sample_temperature is not None:
        config["sample_temperature"] = args.sample_temperature
    out_dir = resolve_path(args.out_dir, config.get("_config_path"))
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size or min(int(config.get("batch_size", 64)), args.num_images)
    all_images: list[torch.Tensor] = []
    generated = 0
    with torch.no_grad():
        while generated < args.num_images:
            current_batch = min(batch_size, args.num_images - generated)
            images = generate_batch(args.model, model, current_batch, config, device)
            save_image_batch(images, out_dir, prefix=args.model, start_index=generated)
            all_images.append(images.cpu())
            generated += current_batch

    images_tensor = torch.cat(all_images, dim=0)
    save_tensor_grid(images_tensor, out_dir / "grid.png")
    print(f"Saved {args.num_images} generated images to {out_dir}")


if __name__ == "__main__":
    main()
