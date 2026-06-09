from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def default_group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(default_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(default_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(default_group_count(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        q = q.view(batch, channels, height * width).transpose(1, 2)
        k = k.view(batch, channels, height * width)
        v = v.view(batch, channels, height * width).transpose(1, 2)
        attention = torch.softmax(torch.bmm(q, k) * (channels**-0.5), dim=-1)
        h = torch.bmm(attention, v).transpose(1, 2).view(batch, channels, height, width)
        return x + self.gamma * self.proj(h)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block1 = ResidualBlock(in_channels, out_channels, dropout)
        self.block2 = ResidualBlock(out_channels, out_channels, dropout)
        self.downsample = nn.Conv2d(out_channels, out_channels, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return self.downsample(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.block1 = ResidualBlock(out_channels, out_channels, dropout)
        self.block2 = ResidualBlock(out_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(self.upsample(x))
        x = self.block1(x)
        return self.block2(x)


class ConvVAE(nn.Module):
    """Residual convolutional VAE for 64x64 RGB images.

    The decoder uses nearest-neighbor upsampling followed by convolutions to
    reduce checkerboard artifacts compared with transposed convolutions.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        channels: int = 3,
        base_channels: int = 64,
        image_size: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.0,
        attention: bool = True,
    ) -> None:
        super().__init__()
        if image_size != 64:
            raise ValueError("This VAE implementation currently supports image_size=64.")
        if len(channel_mults) != 4:
            raise ValueError("For 64x64 images, channel_mults must contain four values.")

        self.latent_dim = latent_dim
        self.channels = channels
        self.base_channels = base_channels
        self.image_size = image_size
        self.channel_mults = channel_mults
        self.dropout = dropout
        self.attention = attention

        stage_channels = [base_channels * mult for mult in channel_mults]
        self.stem = nn.Conv2d(channels, stage_channels[0], 3, 1, 1)
        self.encoder = nn.ModuleList()
        in_channels = stage_channels[0]
        resolution = image_size
        for out_channels in stage_channels:
            self.encoder.append(DownsampleBlock(in_channels, out_channels, dropout))
            resolution //= 2
            if attention and resolution == 16:
                self.encoder.append(AttentionBlock(out_channels))
            in_channels = out_channels

        self.bottleneck = nn.Sequential(
            ResidualBlock(in_channels, in_channels, dropout),
            AttentionBlock(in_channels) if attention else nn.Identity(),
            ResidualBlock(in_channels, in_channels, dropout),
        )
        self.final_channels = in_channels
        self.final_resolution = resolution
        self.flatten_dim = self.final_channels * self.final_resolution * self.final_resolution
        self.encoder_norm = nn.GroupNorm(default_group_count(self.final_channels), self.final_channels)
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder_bottleneck = nn.Sequential(
            ResidualBlock(self.final_channels, self.final_channels, dropout),
            AttentionBlock(self.final_channels) if attention else nn.Identity(),
            ResidualBlock(self.final_channels, self.final_channels, dropout),
        )
        decoder_blocks: list[nn.Module] = []
        reversed_channels = list(reversed(stage_channels))
        in_channels = reversed_channels[0]
        resolution = self.final_resolution
        for out_channels in reversed_channels[1:] + [stage_channels[0]]:
            decoder_blocks.append(UpsampleBlock(in_channels, out_channels, dropout))
            resolution *= 2
            if attention and resolution == 16:
                decoder_blocks.append(AttentionBlock(out_channels))
            in_channels = out_channels
        self.decoder = nn.ModuleList(decoder_blocks)
        self.output_norm = nn.GroupNorm(default_group_count(in_channels), in_channels)
        self.output_conv = nn.Conv2d(in_channels, channels, 3, 1, 1)

    def encode(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.stem(image)
        for layer in self.encoder:
            features = layer(features)
        features = self.bottleneck(features)
        features = F.silu(self.encoder_norm(features)).view(image.size(0), -1)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features).clamp(-8.0, 8.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        features = self.decoder_input(z).view(
            z.size(0),
            self.final_channels,
            self.final_resolution,
            self.final_resolution,
        )
        features = self.decoder_bottleneck(features)
        for layer in self.decoder:
            features = layer(features)
        return torch.tanh(self.output_conv(F.silu(self.output_norm(features))))

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(image)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)


def reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    loss_type: str,
    mse_weight: float,
    l1_weight: float,
) -> torch.Tensor:
    loss_type = loss_type.lower()
    batch_size = target.size(0)
    mse = F.mse_loss(reconstruction, target, reduction="sum") / batch_size
    l1 = F.l1_loss(reconstruction, target, reduction="sum") / batch_size
    if loss_type == "mse":
        return mse
    if loss_type == "l1":
        return l1
    if loss_type in {"l1_mse", "mse_l1"}:
        return mse_weight * mse + l1_weight * l1
    raise ValueError(f"Unsupported VAE reconstruction loss: {loss_type}")


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.0) -> torch.Tensor:
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    return kl_per_dim.sum(dim=1).mean()


def vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    reconstruction_loss_type: str = "l1_mse",
    mse_weight: float = 1.0,
    l1_weight: float = 0.25,
    free_bits: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = reconstruction_loss(
        reconstruction,
        target,
        loss_type=reconstruction_loss_type,
        mse_weight=mse_weight,
        l1_weight=l1_weight,
    )
    kl_loss = kl_divergence(mu, logvar, free_bits=free_bits)
    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


def build_vae_from_config(config: dict) -> ConvVAE:
    return ConvVAE(
        latent_dim=int(config.get("latent_dim", 256)),
        channels=int(config.get("channels", 3)),
        base_channels=int(config.get("base_channels", 64)),
        image_size=int(config.get("image_size", 64)),
        channel_mults=tuple(int(value) for value in config.get("channel_mults", [1, 2, 4, 8])),
        dropout=float(config.get("dropout", 0.0)),
        attention=bool(config.get("attention", True)),
    )

