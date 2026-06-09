from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, optim
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
    parser = argparse.ArgumentParser(description="Train a 64x64 DCGAN.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--resume", default=None, help="Optional training checkpoint to resume from.")
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda, cuda:0, or cpu.")
    return parser.parse_args()


def add_instance_noise(images: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return images
    return (images + torch.randn_like(images) * std).clamp(-1.0, 1.0)


def random_translation(images: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    max_shift = max(1, int(images.size(2) * ratio))
    padded = F.pad(images, (max_shift, max_shift, max_shift, max_shift), mode="reflect")
    out = torch.empty_like(images)
    for index in range(images.size(0)):
        shift_x = torch.randint(0, max_shift * 2 + 1, (1,), device=images.device).item()
        shift_y = torch.randint(0, max_shift * 2 + 1, (1,), device=images.device).item()
        out[index] = padded[index, :, shift_y : shift_y + images.size(2), shift_x : shift_x + images.size(3)]
    return out


def random_cutout(images: torch.Tensor, ratio: float = 0.35) -> torch.Tensor:
    cutout_size = max(1, int(images.size(2) * ratio))
    out = images.clone()
    for index in range(images.size(0)):
        center_y = torch.randint(0, images.size(2), (1,), device=images.device).item()
        center_x = torch.randint(0, images.size(3), (1,), device=images.device).item()
        y0 = max(0, center_y - cutout_size // 2)
        y1 = min(images.size(2), y0 + cutout_size)
        x0 = max(0, center_x - cutout_size // 2)
        x1 = min(images.size(3), x0 + cutout_size)
        out[index, :, y0:y1, x0:x1] = 0
    return out


def random_color(images: torch.Tensor) -> torch.Tensor:
    brightness = torch.rand(images.size(0), 1, 1, 1, device=images.device, dtype=images.dtype) - 0.5
    images = images + brightness
    mean = images.mean(dim=1, keepdim=True)
    contrast = torch.rand(images.size(0), 1, 1, 1, device=images.device, dtype=images.dtype) + 0.5
    images = (images - mean) * contrast + mean
    channel_mean = images.mean(dim=(2, 3), keepdim=True)
    saturation = torch.rand(images.size(0), 1, 1, 1, device=images.device, dtype=images.dtype) * 1.5
    images = (images - channel_mean) * saturation + channel_mean
    return images.clamp(-1.0, 1.0)


def diff_augment(images: torch.Tensor, policy: str) -> torch.Tensor:
    if not policy:
        return images
    out = images
    policies = {name.strip() for name in policy.split(",") if name.strip()}
    if "color" in policies:
        out = random_color(out)
    if "translation" in policies:
        out = random_translation(out)
    if "cutout" in policies:
        out = random_cutout(out)
    return out


def discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    loss_type: str,
    real_targets: torch.Tensor,
    fake_targets: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    if loss_type == "hinge":
        return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
    if loss_type == "wgan_gp":
        return fake_logits.mean() - real_logits.mean()
    if loss_type == "bce":
        return criterion(real_logits, real_targets) + criterion(fake_logits, fake_targets)
    raise ValueError(f"Unsupported GAN loss: {loss_type}")


def generator_loss(fake_logits: torch.Tensor, loss_type: str, targets: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    if loss_type == "hinge":
        return -fake_logits.mean()
    if loss_type == "wgan_gp":
        return -fake_logits.mean()
    if loss_type == "bce":
        return criterion(fake_logits, targets)
    raise ValueError(f"Unsupported GAN loss: {loss_type}")


def gradient_penalty(net_d: nn.Module, real_images: torch.Tensor, fake_images: torch.Tensor) -> torch.Tensor:
    batch_size = real_images.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=real_images.device, dtype=real_images.dtype)
    interpolates = (alpha * real_images + (1.0 - alpha) * fake_images).requires_grad_(True)
    logits = net_d(interpolates)
    gradients = torch.autograd.grad(
        outputs=logits,
        inputs=interpolates,
        grad_outputs=torch.ones_like(logits),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()


def make_ema_model(net_g: nn.Module, enabled: bool) -> nn.Module | None:
    if not enabled:
        return None
    ema_g = copy.deepcopy(net_g)
    ema_g.eval()
    for parameter in ema_g.parameters():
        parameter.requires_grad_(False)
    return ema_g


@torch.no_grad()
def update_ema(ema_g: nn.Module | None, net_g: nn.Module, decay: float) -> None:
    if ema_g is None:
        return
    model_state = net_g.state_dict()
    ema_state = ema_g.state_dict()
    for key, ema_value in ema_state.items():
        model_value = model_state[key]
        if torch.is_floating_point(ema_value):
            ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)
        else:
            ema_value.copy_(model_value)


def checkpoint_payload(
    net_g: nn.Module,
    net_d: nn.Module,
    opt_g: optim.Optimizer,
    opt_d: optim.Optimizer,
    fixed_noise: torch.Tensor,
    config: dict,
    epoch: int,
    ema_g: nn.Module | None = None,
) -> dict:
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
    if ema_g is not None:
        payload["generator_ema_state_dict"] = ema_g.state_dict()
    return payload


def save_checkpoints(
    net_g: nn.Module,
    net_d: nn.Module,
    opt_g: optim.Optimizer,
    opt_d: optim.Optimizer,
    fixed_noise: torch.Tensor,
    config: dict,
    epoch: int,
    save_epoch_copy: bool,
    ema_g: nn.Module | None = None,
) -> None:
    out_dir = checkpoint_dir(config)
    payload = checkpoint_payload(net_g, net_d, opt_g, opt_d, fixed_noise, config, epoch, ema_g)
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
    ema_payload = None
    if ema_g is not None:
        ema_payload = {
            "model": "dcgan_generator_ema",
            "epoch": epoch,
            "config": clean_config(config),
            "state_dict": ema_g.state_dict(),
        }

    torch.save(payload, out_dir / "training_latest.pt")
    torch.save(generator_payload, out_dir / "generator_latest.pt")
    torch.save(discriminator_payload, out_dir / "discriminator_latest.pt")
    if ema_payload is not None:
        torch.save(ema_payload, out_dir / "generator_ema_latest.pt")

    if save_epoch_copy:
        torch.save(payload, out_dir / f"training_epoch_{epoch:04d}.pt")
        torch.save(generator_payload, out_dir / f"generator_epoch_{epoch:04d}.pt")
        torch.save(discriminator_payload, out_dir / f"discriminator_epoch_{epoch:04d}.pt")
        if ema_payload is not None:
            torch.save(ema_payload, out_dir / f"generator_ema_epoch_{epoch:04d}.pt")


def load_resume_checkpoint(
    path: str | Path,
    net_g: nn.Module,
    net_d: nn.Module,
    opt_g: optim.Optimizer,
    opt_d: optim.Optimizer,
    device: torch.device,
    ema_g: nn.Module | None = None,
) -> tuple[int, torch.Tensor | None]:
    checkpoint = torch_load(path, device)
    if "generator_state_dict" in checkpoint:
        net_g.load_state_dict(checkpoint["generator_state_dict"])
        net_d.load_state_dict(checkpoint["discriminator_state_dict"])
        opt_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
        opt_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
        if ema_g is not None:
            ema_g.load_state_dict(checkpoint.get("generator_ema_state_dict", checkpoint["generator_state_dict"]))
        return int(checkpoint.get("epoch", 0)), checkpoint.get("fixed_noise")
    if "state_dict" in checkpoint:
        net_g.load_state_dict(checkpoint["state_dict"])
        if ema_g is not None:
            ema_g.load_state_dict(checkpoint["state_dict"])
        return int(checkpoint.get("epoch", 0)), None
    raise ValueError(f"Unsupported DCGAN checkpoint format: {path}")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    set_seed(int(config.get("seed", 42)))

    device = get_device(args.device)
    amp_enabled = use_mixed_precision(config, device)
    dataset, loader = build_dataloader(config, train=True)

    net_g = build_generator_from_config(config).to(device)
    net_d = build_discriminator_from_config(config).to(device)
    net_g.apply(weights_init)
    net_d.apply(weights_init)
    ema_decay = float(config.get("ema_decay", 0.0))
    ema_g = make_ema_model(net_g, enabled=ema_decay > 0)

    opt_g = optim.Adam(
        net_g.parameters(),
        lr=float(config.get("lr_g", 0.0002)),
        betas=(float(config.get("beta1", 0.5)), float(config.get("beta2", 0.999))),
    )
    opt_d = optim.Adam(
        net_d.parameters(),
        lr=float(config.get("lr_d", 0.0002)),
        betas=(float(config.get("beta1", 0.5)), float(config.get("beta2", 0.999))),
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = make_grad_scaler(device, amp_enabled)

    sample_count = int(config.get("num_sample_images", 64))
    latent_dim = int(config.get("latent_dim", 100))
    fixed_noise = torch.randn(sample_count, latent_dim, 1, 1, device=device)
    start_epoch = 0
    if args.resume:
        start_epoch, resumed_noise = load_resume_checkpoint(args.resume, net_g, net_d, opt_g, opt_d, device, ema_g)
        if resumed_noise is not None:
            fixed_noise = resumed_noise.to(device)

    ckpt_dir = checkpoint_dir(config)
    samples = sample_dir(config)
    copy_config(config, ckpt_dir / "config.yaml")

    print(f"Dataset: {dataset.root} ({len(dataset)} images)")
    print(f"Device: {device}; AMP: {amp_enabled}")
    print(f"Generator parameters: {count_parameters(net_g):,}")
    print(f"Discriminator parameters: {count_parameters(net_d):,}")
    if ema_g is not None:
        print(f"EMA generator enabled with decay={ema_decay}")

    epochs = int(config.get("epochs", 100))
    sample_every = int(config.get("sample_every", 5))
    checkpoint_every = int(config.get("checkpoint_every", 10))
    real_smoothing = float(config.get("real_label_smoothing", 0.9))
    instance_noise_std = float(config.get("instance_noise_std", 0.0))
    gan_loss = str(config.get("gan_loss", "bce")).lower()
    augment_policy = str(config.get("diff_augment_policy", "")) if bool(config.get("diff_augment", False)) else ""
    discriminator_steps = max(1, int(config.get("discriminator_steps", 1)))
    gradient_penalty_weight = float(config.get("gradient_penalty_weight", 10.0))
    if gan_loss == "wgan_gp":
        print(
            f"WGAN-GP enabled: {discriminator_steps} critic step(s), "
            f"gradient penalty weight={gradient_penalty_weight}. "
            "Logged D(real)/D(fake) are raw critic scores."
        )
    elif gan_loss == "hinge":
        print("Hinge loss enabled. Logged D(real)/D(fake) are raw discriminator scores.")
    metrics_path = ckpt_dir / "metrics.csv"

    for epoch in range(start_epoch + 1, epochs + 1):
        epoch_start = time.time()
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

            for _ in range(discriminator_steps):
                opt_d.zero_grad(set_to_none=True)
                if gan_loss == "wgan_gp":
                    noisy_real = add_instance_noise(real_images, instance_noise_std)
                    real_input = diff_augment(noisy_real, augment_policy)
                    with torch.no_grad():
                        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
                        fake_images = net_g(noise)
                    noisy_fake = add_instance_noise(fake_images, instance_noise_std)
                    fake_input = diff_augment(noisy_fake, augment_policy)
                    real_logits = net_d(real_input.float())
                    fake_logits = net_d(fake_input.float())
                    gp = gradient_penalty(net_d, real_input.float(), fake_input.float())
                    loss_d = discriminator_loss(real_logits, fake_logits, gan_loss, real_targets, fake_targets, criterion)
                    loss_d = loss_d + gradient_penalty_weight * gp
                    loss_d.backward()
                    opt_d.step()
                else:
                    with autocast_context(device, amp_enabled):
                        noisy_real = add_instance_noise(real_images, instance_noise_std)
                        real_input = diff_augment(noisy_real, augment_policy)
                        real_logits = net_d(real_input)

                        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
                        fake_images = net_g(noise)
                        noisy_fake = add_instance_noise(fake_images.detach(), instance_noise_std)
                        fake_input = diff_augment(noisy_fake, augment_policy)
                        fake_logits = net_d(fake_input)
                        loss_d = discriminator_loss(real_logits, fake_logits, gan_loss, real_targets, fake_targets, criterion)
                    scaler.scale(loss_d).backward()
                    scaler.step(opt_d)
                    scaler.update()

            opt_g.zero_grad(set_to_none=True)
            with autocast_context(device, amp_enabled and gan_loss != "wgan_gp"):
                noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
                generated = net_g(noise)
                generator_input = diff_augment(generated, augment_policy)
                generator_logits = net_d(generator_input)
                loss_g = generator_loss(generator_logits, gan_loss, torch.ones(batch_size, device=device), criterion)
            if gan_loss == "wgan_gp":
                loss_g.backward()
                opt_g.step()
            else:
                scaler.scale(loss_g).backward()
                scaler.step(opt_g)
                scaler.update()
            update_ema(ema_g, net_g, ema_decay)

            with torch.no_grad():
                if gan_loss in {"wgan_gp", "hinge"}:
                    real_score = real_logits.mean().item()
                    fake_score = fake_logits.mean().item()
                else:
                    real_score = torch.sigmoid(real_logits).mean().item()
                    fake_score = torch.sigmoid(fake_logits).mean().item()
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
            sample_model = ema_g if ema_g is not None else net_g
            sample_model.eval()
            with torch.no_grad():
                samples_tensor = sample_model(fixed_noise)
            save_tensor_grid(samples_tensor, samples / f"epoch_{epoch:04d}.png")
            save_tensor_grid(samples_tensor, samples / "latest.png")

        save_epoch_copy = epoch % checkpoint_every == 0 or epoch == epochs
        save_checkpoints(net_g, net_d, opt_g, opt_d, fixed_noise, config, epoch, save_epoch_copy, ema_g)


if __name__ == "__main__":
    main()
