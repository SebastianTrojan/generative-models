from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from .data import build_dataloader
from .models.dcgan import build_discriminator_from_config, build_generator_from_config, weights_init
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


METRIC_FIELDS = [
    "epoch",
    "generator_loss",
    "discriminator_loss",
    "discriminator_real_score",
    "discriminator_fake_score",
    "epoch_time_sec",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the 64x64 DCGAN.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--resume", default=None, help="Optional training checkpoint to resume from.")
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda, cuda:0, or cpu.")
    return parser.parse_args()


def add_instance_noise(images: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return images
    return (images + torch.randn_like(images) * std).clamp(-1.0, 1.0)


def instance_noise_for_epoch(base_std: float, decay_epochs: int, epoch: int) -> float:
    if base_std <= 0:
        return 0.0
    if decay_epochs <= 0:
        return base_std
    progress = min(1.0, max(0.0, (epoch - 1) / decay_epochs))
    return base_std * (1.0 - progress)


def discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    real_targets: torch.Tensor,
    fake_targets: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * (
        F.mse_loss(real_logits, real_targets)
        + F.mse_loss(fake_logits, fake_targets)
    )


def generator_loss(fake_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return 0.5 * F.mse_loss(fake_logits, targets)


def ensure_finite(tensor: torch.Tensor, name: str, epoch: int) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"Non-finite {name} detected at epoch {epoch}; stop this run and lower learning rate.")


def save_checkpoints(
    net_g,
    net_d,
    opt_g: optim.Optimizer,
    opt_d: optim.Optimizer,
    fixed_noise: torch.Tensor,
    config: dict,
    epoch: int,
    save_epoch_copy: bool,
) -> None:
    out_dir = checkpoint_dir(config)
    payload = {
        "model": "dcgan",
        "epoch": epoch,
        "config": clean_config(config),
        "generator_state_dict": net_g.state_dict(),
        "discriminator_state_dict": net_d.state_dict(),
        "optimizer_g_state_dict": opt_g.state_dict(),
        "optimizer_d_state_dict": opt_d.state_dict(),
        "fixed_noise": fixed_noise.detach().cpu(),
    }
    generator_payload = {
        "model": "dcgan_generator",
        "epoch": epoch,
        "config": clean_config(config),
        "state_dict": net_g.state_dict(),
    }
    discriminator_payload = {
        "model": "dcgan_discriminator",
        "epoch": epoch,
        "config": clean_config(config),
        "state_dict": net_d.state_dict(),
    }

    torch.save(payload, out_dir / "training_latest.pt")
    torch.save(generator_payload, out_dir / "generator_latest.pt")
    torch.save(discriminator_payload, out_dir / "discriminator_latest.pt")

    if save_epoch_copy:
        torch.save(payload, out_dir / f"training_epoch_{epoch:04d}.pt")
        torch.save(generator_payload, out_dir / f"generator_epoch_{epoch:04d}.pt")
        torch.save(discriminator_payload, out_dir / f"discriminator_epoch_{epoch:04d}.pt")


def load_resume_checkpoint(
    path: str | Path,
    net_g,
    net_d,
    opt_g: optim.Optimizer,
    opt_d: optim.Optimizer,
    device: torch.device,
) -> tuple[int, torch.Tensor | None]:
    checkpoint = torch_load(path, device)
    if "generator_state_dict" not in checkpoint:
        raise ValueError(f"Resume requires a training checkpoint with generator/discriminator state: {path}")
    net_g.load_state_dict(checkpoint["generator_state_dict"])
    net_d.load_state_dict(checkpoint["discriminator_state_dict"])
    opt_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
    opt_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
    return int(checkpoint.get("epoch", 0)), checkpoint.get("fixed_noise")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    set_seed(int(config.get("seed", 42)))

    gan_loss = str(config.get("gan_loss", "lsgan")).lower()
    if gan_loss != "lsgan":
        raise ValueError("Only gan_loss: lsgan is supported by the DCGAN trainer.")

    device = get_device(args.device)
    amp_enabled = use_mixed_precision(config, device)
    dataset, loader = build_dataloader(config, train=True)

    net_g = build_generator_from_config(config).to(device)
    net_d = build_discriminator_from_config(config).to(device)
    net_g.apply(weights_init)
    net_d.apply(weights_init)

    opt_g = optim.Adam(
        net_g.parameters(),
        lr=float(config.get("lr_g", 0.0002)),
        betas=(float(config.get("beta1", 0.5)), float(config.get("beta2", 0.999))),
    )
    opt_d = optim.Adam(
        net_d.parameters(),
        lr=float(config.get("lr_d", 0.00005)),
        betas=(float(config.get("beta1", 0.5)), float(config.get("beta2", 0.999))),
    )
    scaler = make_grad_scaler(device, amp_enabled)

    sample_count = int(config.get("num_sample_images", 64))
    latent_dim = int(config.get("latent_dim", 100))
    fixed_noise = torch.randn(sample_count, latent_dim, 1, 1, device=device)
    start_epoch = 0
    if args.resume:
        start_epoch, resumed_noise = load_resume_checkpoint(args.resume, net_g, net_d, opt_g, opt_d, device)
        if resumed_noise is not None:
            fixed_noise = resumed_noise.to(device)

    ckpt_dir = checkpoint_dir(config)
    samples = sample_dir(config)
    copy_config(config, ckpt_dir / "config.yaml")

    print(f"Dataset: {dataset.root} ({len(dataset)} images)")
    print(f"Device: {device}; AMP: {amp_enabled}")
    print(f"Generator parameters: {count_parameters(net_g):,}")
    print(f"Discriminator parameters: {count_parameters(net_d):,}")
    print("LSGAN loss enabled. Logged D(real)/D(fake) are raw discriminator scores.")

    epochs = int(config.get("epochs", 100))
    sample_every = int(config.get("sample_every", 5))
    checkpoint_every = int(config.get("checkpoint_every", 10))
    real_smoothing = float(config.get("real_label_smoothing", 0.9))
    instance_noise_std = float(config.get("instance_noise_std", 0.0))
    instance_noise_decay_epochs = int(config.get("instance_noise_decay_epochs", 0))
    grad_clip_g = float(config.get("grad_clip_g", 0.0))
    grad_clip_d = float(config.get("grad_clip_d", 0.0))
    fail_on_nonfinite = bool(config.get("fail_on_nonfinite", True))
    metrics_path = ckpt_dir / "metrics.csv"

    for epoch in range(start_epoch + 1, epochs + 1):
        epoch_start = time.time()
        current_instance_noise = instance_noise_for_epoch(instance_noise_std, instance_noise_decay_epochs, epoch)
        net_g.train()
        net_d.train()
        running_g = 0.0
        running_d = 0.0
        running_real_score = 0.0
        running_fake_score = 0.0
        num_batches = 0

        progress = tqdm(loader, desc=f"DCGAN epoch {epoch}/{epochs}", leave=False)
        for real_images in progress:
            real_images = real_images.to(device, non_blocking=True)
            batch_size = real_images.size(0)
            real_targets = torch.full((batch_size,), real_smoothing, device=device)
            fake_targets = torch.zeros(batch_size, device=device)

            opt_d.zero_grad(set_to_none=True)
            with autocast_context(device, amp_enabled):
                real_input = add_instance_noise(real_images, current_instance_noise)
                real_logits = net_d(real_input)

                with torch.no_grad():
                    noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
                    fake_images = net_g(noise)
                fake_input = add_instance_noise(fake_images, current_instance_noise)
                fake_logits = net_d(fake_input)
                loss_d = discriminator_loss(real_logits, fake_logits, real_targets, fake_targets)
            if fail_on_nonfinite:
                ensure_finite(loss_d, "discriminator loss", epoch)
            scaler.scale(loss_d).backward()
            if grad_clip_d > 0:
                scaler.unscale_(opt_d)
                torch.nn.utils.clip_grad_norm_(net_d.parameters(), grad_clip_d)
            scaler.step(opt_d)
            scaler.update()

            opt_g.zero_grad(set_to_none=True)
            with autocast_context(device, amp_enabled):
                noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
                generated = net_g(noise)
                generator_logits = net_d(generated)
                loss_g = generator_loss(generator_logits, torch.ones(batch_size, device=device))
            if fail_on_nonfinite:
                ensure_finite(loss_g, "generator loss", epoch)
            scaler.scale(loss_g).backward()
            if grad_clip_g > 0:
                scaler.unscale_(opt_g)
                torch.nn.utils.clip_grad_norm_(net_g.parameters(), grad_clip_g)
            scaler.step(opt_g)
            scaler.update()

            with torch.no_grad():
                real_score = real_logits.mean().item()
                fake_score = fake_logits.mean().item()
            running_g += loss_g.item()
            running_d += loss_d.item()
            running_real_score += real_score
            running_fake_score += fake_score
            num_batches += 1
            progress.set_postfix(
                loss_g=f"{loss_g.item():.3f}",
                loss_d=f"{loss_d.item():.3f}",
                d_real=f"{real_score:.3f}",
                d_fake=f"{fake_score:.3f}",
            )

        avg_g = running_g / max(1, num_batches)
        avg_d = running_d / max(1, num_batches)
        avg_real_score = running_real_score / max(1, num_batches)
        avg_fake_score = running_fake_score / max(1, num_batches)
        epoch_time = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "generator_loss": f"{avg_g:.6f}",
            "discriminator_loss": f"{avg_d:.6f}",
            "discriminator_real_score": f"{avg_real_score:.6f}",
            "discriminator_fake_score": f"{avg_fake_score:.6f}",
            "epoch_time_sec": f"{epoch_time:.2f}",
        }
        append_metrics_csv(metrics_path, METRIC_FIELDS, row)
        print(
            f"Epoch {epoch:04d}: G={avg_g:.4f} D={avg_d:.4f} "
            f"D(real)={avg_real_score:.3f} D(fake)={avg_fake_score:.3f} "
            f"time={epoch_time:.1f}s"
        )

        if epoch == 1 or epoch % sample_every == 0 or epoch == epochs:
            net_g.eval()
            with torch.no_grad():
                samples_tensor = net_g(fixed_noise)
            save_tensor_grid(samples_tensor, samples / f"epoch_{epoch:04d}.png")
            save_tensor_grid(samples_tensor, samples / "latest.png")

        save_checkpoint_copy = epoch % checkpoint_every == 0 or epoch == epochs
        save_checkpoints(net_g, net_d, opt_g, opt_d, fixed_noise, config, epoch, save_checkpoint_copy)


if __name__ == "__main__":
    main()
