from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import optim
from tqdm import tqdm

from .data import build_dataloader
from .models.ddpm import build_denoiser_from_config, build_diffusion_from_config
from .utils import (
    append_metrics_csv,
    autocast_context,
    checkpoint_dir,
    clean_config,
    copy_config,
    count_parameters,
    get_device,
    load_yaml,
    make_grad_scaler,
    sample_dir,
    save_tensor_grid,
    set_seed,
    torch_load,
    use_mixed_precision,
)


METRIC_FIELDS = ["epoch", "noise_prediction_loss", "epoch_time_sec"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact 64x64 DDPM.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--resume", default=None, help="Optional training checkpoint to resume from.")
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda, cuda:0, or cpu.")
    return parser.parse_args()


def save_checkpoint(model, optimizer, config: dict, epoch: int, save_epoch_copy: bool) -> None:
    out_dir = checkpoint_dir(config)
    payload = {
        "model": "ddpm",
        "epoch": epoch,
        "config": clean_config(config),
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    model_payload = {
        "model": "ddpm",
        "epoch": epoch,
        "config": clean_config(config),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, out_dir / "training_latest.pt")
    torch.save(model_payload, out_dir / "model_latest.pt")
    if save_epoch_copy:
        torch.save(payload, out_dir / f"training_epoch_{epoch:04d}.pt")
        torch.save(model_payload, out_dir / f"model_epoch_{epoch:04d}.pt")


def load_resume_checkpoint(path: str | Path, model, optimizer, device: torch.device) -> int:
    checkpoint = torch_load(path, device)
    if "state_dict" not in checkpoint:
        raise ValueError(f"Unsupported DDPM checkpoint format: {path}")
    model.load_state_dict(checkpoint["state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("epoch", 0))


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    set_seed(int(config.get("seed", 42)))

    device = get_device(args.device)
    amp_enabled = use_mixed_precision(config, device)
    dataset, loader = build_dataloader(config, train=True)

    model = build_denoiser_from_config(config).to(device)
    diffusion = build_diffusion_from_config(config, device=device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", 0.0002)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    scaler = make_grad_scaler(device, amp_enabled)

    ckpt_dir = checkpoint_dir(config)
    samples = sample_dir(config)
    copy_config(config, ckpt_dir / "config.yaml")

    start_epoch = 0
    if args.resume:
        start_epoch = load_resume_checkpoint(args.resume, model, optimizer, device)

    print(f"Dataset: {dataset.root} ({len(dataset)} images)")
    print(f"Device: {device}; AMP: {amp_enabled}")
    print(f"DDPM denoiser parameters: {count_parameters(model):,}")
    print(f"Diffusion timesteps: {diffusion.timesteps}")

    epochs = int(config.get("epochs", 100))
    sample_every = int(config.get("sample_every", 5))
    checkpoint_every = int(config.get("checkpoint_every", 10))
    num_sample_images = int(config.get("num_sample_images", 16))
    grad_clip = float(config.get("grad_clip", 1.0))
    metrics_path = ckpt_dir / "metrics.csv"

    for epoch in range(start_epoch + 1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        num_batches = 0

        progress = tqdm(loader, desc=f"DDPM epoch {epoch}/{epochs}", leave=False)
        for images in progress:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp_enabled):
                loss = diffusion.training_loss(model, images)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            num_batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / max(1, num_batches)
        epoch_time = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "noise_prediction_loss": f"{avg_loss:.6f}",
            "epoch_time_sec": f"{epoch_time:.2f}",
        }
        append_metrics_csv(metrics_path, METRIC_FIELDS, row)
        print(f"Epoch {epoch:04d}: noise_loss={avg_loss:.6f} time={epoch_time:.1f}s")

        if epoch == 1 or epoch % sample_every == 0 or epoch == epochs:
            model.eval()
            shape = (
                num_sample_images,
                int(config.get("channels", 3)),
                int(config.get("image_size", 64)),
                int(config.get("image_size", 64)),
            )
            sample_start = time.time()
            with torch.no_grad():
                generated = diffusion.p_sample_loop(model, shape, device=device, progress=True)
            save_tensor_grid(generated, samples / f"epoch_{epoch:04d}.png")
            save_tensor_grid(generated, samples / "latest.png")
            print(f"Saved DDPM samples in {time.time() - sample_start:.1f}s")

        save_checkpoint(
            model,
            optimizer,
            config,
            epoch,
            save_epoch_copy=epoch % checkpoint_every == 0 or epoch == epochs,
        )


if __name__ == "__main__":
    main()
