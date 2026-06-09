from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import optim
from tqdm import tqdm

from .data import build_dataloader
from .models.vae import build_vae_from_config, vae_loss
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


METRIC_FIELDS = ["epoch", "total_loss", "reconstruction_loss", "kl_loss", "beta", "epoch_time_sec"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 64x64 residual convolutional VAE.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--resume", default=None, help="Optional training checkpoint to resume from.")
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda, cuda:0, or cpu.")
    return parser.parse_args()


def save_checkpoint(model, optimizer, config: dict, epoch: int, save_epoch_copy: bool) -> None:
    out_dir = checkpoint_dir(config)
    payload = {
        "model": "vae",
        "epoch": epoch,
        "config": clean_config(config),
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    model_payload = {
        "model": "vae",
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
        raise ValueError(f"Unsupported VAE checkpoint format: {path}")
    model.load_state_dict(checkpoint["state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("epoch", 0))


def beta_for_epoch(config: dict, epoch: int) -> float:
    target_beta = float(config.get("beta", 1.0))
    start_beta = float(config.get("beta_start", 0.0))
    warmup_epochs = int(config.get("kl_warmup_epochs", 0))
    if warmup_epochs <= 0:
        return target_beta
    progress = min(1.0, max(0.0, epoch / warmup_epochs))
    return start_beta + (target_beta - start_beta) * progress


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    set_seed(int(config.get("seed", 42)))
    device = get_device(args.device)
    amp_enabled = use_mixed_precision(config, device)

    dataset, loader = build_dataloader(config, train=True)
    model = build_vae_from_config(config).to(device)
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

    fixed_batch = next(iter(loader))
    fixed_images = fixed_batch[: int(config.get("num_sample_images", 64))].to(device)

    print(f"Dataset: {dataset.root} ({len(dataset)} images)")
    print(f"Device: {device}; AMP: {amp_enabled}")
    print(f"VAE parameters: {count_parameters(model):,}")

    epochs = int(config.get("epochs", 50))
    sample_every = int(config.get("sample_every", 5))
    checkpoint_every = int(config.get("checkpoint_every", 10))
    reconstruction_loss_type = str(config.get("reconstruction_loss", "l1_mse"))
    mse_weight = float(config.get("mse_weight", 1.0))
    l1_weight = float(config.get("l1_weight", 0.25))
    free_bits = float(config.get("free_bits", 0.0))
    grad_clip = float(config.get("grad_clip", 0.0))
    metrics_path = ckpt_dir / "metrics.csv"

    for epoch in range(start_epoch + 1, epochs + 1):
        epoch_start = time.time()
        beta = beta_for_epoch(config, epoch)
        model.train()
        running_total = 0.0
        running_recon = 0.0
        running_kl = 0.0
        num_batches = 0

        progress = tqdm(loader, desc=f"VAE epoch {epoch}/{epochs}", leave=False)
        for images in progress:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp_enabled):
                reconstruction, mu, logvar = model(images)
                loss, recon_loss, kl_loss = vae_loss(
                    reconstruction,
                    images,
                    mu,
                    logvar,
                    beta=beta,
                    reconstruction_loss_type=reconstruction_loss_type,
                    mse_weight=mse_weight,
                    l1_weight=l1_weight,
                    free_bits=free_bits,
                )
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_total += loss.item()
            running_recon += recon_loss.item()
            running_kl += kl_loss.item()
            num_batches += 1
            progress.set_postfix(
                loss=f"{loss.item():.2f}",
                recon=f"{recon_loss.item():.2f}",
                kl=f"{kl_loss.item():.2f}",
            )

        avg_total = running_total / max(1, num_batches)
        avg_recon = running_recon / max(1, num_batches)
        avg_kl = running_kl / max(1, num_batches)
        epoch_time = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "total_loss": f"{avg_total:.6f}",
            "reconstruction_loss": f"{avg_recon:.6f}",
            "kl_loss": f"{avg_kl:.6f}",
            "beta": f"{beta:.6f}",
            "epoch_time_sec": f"{epoch_time:.2f}",
        }
        append_metrics_csv(metrics_path, METRIC_FIELDS, row)
        print(
            f"Epoch {epoch:04d}: loss={avg_total:.4f} recon={avg_recon:.4f} "
            f"kl={avg_kl:.4f} beta={beta:.4f} time={epoch_time:.1f}s"
        )

        if epoch == 1 or epoch % sample_every == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                reconstruction, _, _ = model(fixed_images)
                random_samples = model.sample(int(config.get("num_sample_images", 64)), device)
            recon_grid = torch.cat([fixed_images, reconstruction], dim=0)
            save_tensor_grid(recon_grid, samples / f"reconstruction_epoch_{epoch:04d}.png")
            save_tensor_grid(random_samples, samples / f"samples_epoch_{epoch:04d}.png")
            save_tensor_grid(recon_grid, samples / "reconstruction_latest.png")
            save_tensor_grid(random_samples, samples / "samples_latest.png")

        save_checkpoint(
            model,
            optimizer,
            config,
            epoch,
            save_epoch_copy=epoch % checkpoint_every == 0 or epoch == epochs,
        )


if __name__ == "__main__":
    main()
