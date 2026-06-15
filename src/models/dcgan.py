from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import spectral_norm as apply_spectral_norm


def weights_init(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if classname.find("Conv") != -1 or classname.find("Linear") != -1:
        weight = getattr(module, "weight_orig", getattr(module, "weight", None))
        if weight is not None:
            nn.init.normal_(weight.data, 0.0, 0.02)
        bias = getattr(module, "bias", None)
        if bias is not None:
            nn.init.constant_(bias.data, 0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


def maybe_spectral_norm(layer: nn.Module, enabled: bool) -> nn.Module:
    return apply_spectral_norm(layer) if enabled else layer


class MinibatchStdDev(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(0) <= 1:
            std = x.new_zeros(x.size(0), 1, x.size(2), x.size(3))
        else:
            value = x.float().std(dim=0, unbiased=False).mean().to(dtype=x.dtype)
            std = value.expand(x.size(0), 1, x.size(2), x.size(3))
        return torch.cat([x, std], dim=1)


class DCGANGenerator(nn.Module):
    """64x64 DCGAN generator."""

    def __init__(
        self,
        latent_dim: int = 100,
        channels: int = 3,
        feature_maps: int = 64,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        if image_size != 64:
            raise ValueError("DCGANGenerator currently supports image_size=64.")
        self.latent_dim = latent_dim
        self.channels = channels
        self.feature_maps = feature_maps
        self.image_size = image_size

        ngf = feature_maps
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DCGANDiscriminator(nn.Module):
    """64x64 DCGAN discriminator."""

    def __init__(
        self,
        channels: int = 3,
        feature_maps: int = 64,
        image_size: int = 64,
        spectral_norm: bool = False,
        minibatch_stddev: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if image_size != 64:
            raise ValueError("DCGANDiscriminator currently supports image_size=64.")
        ndf = feature_maps

        def conv(in_channels: int, out_channels: int, use_batchnorm: bool) -> list[nn.Module]:
            layers: list[nn.Module] = [
                maybe_spectral_norm(
                    nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                    spectral_norm,
                )
            ]
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            return layers

        final_channels = ndf * 8 + (1 if minibatch_stddev else 0)
        self.net = nn.Sequential(
            *conv(channels, ndf, use_batchnorm=False),
            *conv(ndf, ndf * 2, use_batchnorm=True),
            *conv(ndf * 2, ndf * 4, use_batchnorm=True),
            *conv(ndf * 4, ndf * 8, use_batchnorm=True),
            *([MinibatchStdDev()] if minibatch_stddev else []),
            maybe_spectral_norm(nn.Conv2d(final_channels, 1, 4, 1, 0, bias=False), spectral_norm),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image).view(-1)


def build_generator_from_config(config: dict) -> DCGANGenerator:
    architecture = str(config.get("architecture", "dcgan")).lower()
    if architecture != "dcgan":
        raise ValueError(f"Only architecture='dcgan' is supported, got: {architecture}")
    return DCGANGenerator(
        latent_dim=int(config.get("latent_dim", 100)),
        channels=int(config.get("channels", 3)),
        feature_maps=int(config.get("generator_features", 64)),
        image_size=int(config.get("image_size", 64)),
    )


def build_discriminator_from_config(config: dict) -> DCGANDiscriminator:
    architecture = str(config.get("architecture", "dcgan")).lower()
    if architecture != "dcgan":
        raise ValueError(f"Only architecture='dcgan' is supported, got: {architecture}")
    return DCGANDiscriminator(
        channels=int(config.get("channels", 3)),
        feature_maps=int(config.get("discriminator_features", 64)),
        image_size=int(config.get("image_size", 64)),
        spectral_norm=bool(config.get("spectral_norm", False)),
        minibatch_stddev=bool(config.get("minibatch_stddev", False)),
        dropout=float(config.get("discriminator_dropout", 0.0)),
    )
