from __future__ import annotations

import torch
import torch.nn.functional as F
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


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int, spectral_norm: bool = False) -> None:
        super().__init__()
        hidden = max(1, channels // 8)
        self.query = maybe_spectral_norm(nn.Conv2d(channels, hidden, 1), spectral_norm)
        self.key = maybe_spectral_norm(nn.Conv2d(channels, hidden, 1), spectral_norm)
        self.value = maybe_spectral_norm(nn.Conv2d(channels, channels, 1), spectral_norm)
        self.out = maybe_spectral_norm(nn.Conv2d(channels, channels, 1), spectral_norm)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        positions = height * width
        query = self.query(x).view(batch, -1, positions).transpose(1, 2)
        key = self.key(x).view(batch, -1, positions)
        attention = torch.softmax(torch.bmm(query, key), dim=-1)
        value = self.value(x).view(batch, channels, positions)
        out = torch.bmm(value, attention.transpose(1, 2)).view(batch, channels, height, width)
        return x + self.gamma * self.out(out)


class MinibatchStdDev(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(0) <= 1:
            std = x.new_zeros(x.size(0), 1, x.size(2), x.size(3))
        else:
            value = x.float().std(dim=0, unbiased=False).mean().to(dtype=x.dtype)
            std = value.expand(x.size(0), 1, x.size(2), x.size(3))
        return torch.cat([x, std], dim=1)


class ResidualUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.interpolate(x, scale_factor=2, mode="nearest")
        residual = self.skip(residual)
        h = F.relu(self.bn1(x), inplace=True)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.conv1(h)
        h = self.conv2(F.relu(self.bn2(h), inplace=True))
        return h + residual


class ResidualDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, first_block: bool = False) -> None:
        super().__init__()
        self.first_block = first_block
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.avg_pool2d(self.skip(x), 2)
        h = x if self.first_block else F.leaky_relu(x, 0.2, inplace=False)
        h = self.conv1(h)
        h = F.leaky_relu(h, 0.2, inplace=True)
        h = self.conv2(h)
        h = F.avg_pool2d(h, 2)
        return h + residual


class ResidualGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        channels: int = 3,
        feature_maps: int = 64,
        image_size: int = 64,
        attention_resolutions: set[int] | None = None,
    ) -> None:
        super().__init__()
        if image_size != 64:
            raise ValueError("ResidualGenerator currently supports image_size=64.")
        attention_resolutions = attention_resolutions or set()
        self.latent_dim = latent_dim
        self.channels = channels
        self.feature_maps = feature_maps
        self.image_size = image_size
        ngf = feature_maps
        self.fc = nn.Linear(latent_dim, ngf * 8 * 4 * 4)
        self.net = nn.Sequential(
            ResidualUpBlock(ngf * 8, ngf * 8),
            *maybe_attention(ngf * 8, 8, attention_resolutions),
            ResidualUpBlock(ngf * 8, ngf * 4),
            *maybe_attention(ngf * 4, 16, attention_resolutions),
            ResidualUpBlock(ngf * 4, ngf * 2),
            *maybe_attention(ngf * 2, 32, attention_resolutions),
            ResidualUpBlock(ngf * 2, ngf),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, channels, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 4:
            z = z.flatten(1)
        h = self.fc(z).view(z.size(0), self.feature_maps * 8, 4, 4)
        return self.net(h)


class ResidualCritic(nn.Module):
    def __init__(
        self,
        channels: int = 3,
        feature_maps: int = 64,
        image_size: int = 64,
        attention_resolutions: set[int] | None = None,
    ) -> None:
        super().__init__()
        if image_size != 64:
            raise ValueError("ResidualCritic currently supports image_size=64.")
        attention_resolutions = attention_resolutions or set()
        ndf = feature_maps
        self.net = nn.Sequential(
            ResidualDownBlock(channels, ndf, first_block=True),
            *maybe_attention(ndf, 32, attention_resolutions),
            ResidualDownBlock(ndf, ndf * 2),
            *maybe_attention(ndf * 2, 16, attention_resolutions),
            ResidualDownBlock(ndf * 2, ndf * 4),
            *maybe_attention(ndf * 4, 8, attention_resolutions),
            ResidualDownBlock(ndf * 4, ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.linear = nn.Linear(ndf * 8 * 4 * 4, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        h = self.net(image).view(image.size(0), -1)
        return self.linear(h).view(-1)


def parse_resolutions(value) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(item) for item in value}


def maybe_attention(channels: int, resolution: int, attention_resolutions: set[int], spectral_norm: bool = False) -> list[nn.Module]:
    if resolution in attention_resolutions:
        return [SelfAttention2d(channels, spectral_norm=spectral_norm)]
    return []


def generator_refinement(channels: int, blocks: int) -> list[nn.Module]:
    layers: list[nn.Module] = []
    for _ in range(blocks):
        layers.extend(
            [
                nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
            ]
        )
    return layers


def discriminator_refinement(channels: int, blocks: int, spectral_norm: bool, use_batchnorm: bool) -> list[nn.Module]:
    layers: list[nn.Module] = []
    for _ in range(blocks):
        layers.append(
            maybe_spectral_norm(nn.Conv2d(channels, channels, 3, 1, 1, bias=False), spectral_norm)
        )
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


def normalization(channels: int, enabled: bool) -> list[nn.Module]:
    return [nn.BatchNorm2d(channels)] if enabled else []


def upsample_conv_block(in_channels: int, out_channels: int, refine_blocks: int) -> list[nn.Module]:
    return [
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
        *generator_refinement(out_channels, refine_blocks),
    ]


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 100,
        channels: int = 3,
        feature_maps: int = 64,
        image_size: int = 64,
        refine_blocks: int = 0,
        attention_resolutions: set[int] | None = None,
        upsample_mode: str = "transpose",
    ) -> None:
        super().__init__()
        if image_size != 64:
            raise ValueError("This compact DCGAN implementation currently supports image_size=64.")
        self.latent_dim = latent_dim
        self.channels = channels
        self.feature_maps = feature_maps
        self.image_size = image_size
        self.refine_blocks = refine_blocks
        self.attention_resolutions = attention_resolutions or set()
        self.upsample_mode = upsample_mode
        ngf = feature_maps
        if upsample_mode == "transpose":
            self.net = nn.Sequential(
                nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                *generator_refinement(ngf * 8, refine_blocks),
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                *generator_refinement(ngf * 4, refine_blocks),
                *maybe_attention(ngf * 4, 8, self.attention_resolutions),
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                *generator_refinement(ngf * 2, refine_blocks),
                *maybe_attention(ngf * 2, 16, self.attention_resolutions),
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                *generator_refinement(ngf, refine_blocks),
                *maybe_attention(ngf, 32, self.attention_resolutions),
                nn.ConvTranspose2d(ngf, channels, 4, 2, 1, bias=False),
                nn.Tanh(),
            )
        elif upsample_mode == "nearest_conv":
            self.net = nn.Sequential(
                nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                *generator_refinement(ngf * 8, refine_blocks),
                *upsample_conv_block(ngf * 8, ngf * 4, refine_blocks),
                *maybe_attention(ngf * 4, 8, self.attention_resolutions),
                *upsample_conv_block(ngf * 4, ngf * 2, refine_blocks),
                *maybe_attention(ngf * 2, 16, self.attention_resolutions),
                *upsample_conv_block(ngf * 2, ngf, refine_blocks),
                *maybe_attention(ngf, 32, self.attention_resolutions),
                *upsample_conv_block(ngf, ngf, refine_blocks),
                nn.Conv2d(ngf, channels, 3, 1, 1, bias=False),
                nn.Tanh(),
            )
        else:
            raise ValueError(f"Unknown generator upsample mode: {upsample_mode}")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(
        self,
        channels: int = 3,
        feature_maps: int = 64,
        image_size: int = 64,
        spectral_norm: bool = False,
        refine_blocks: int = 0,
        attention_resolutions: set[int] | None = None,
        use_batchnorm: bool = True,
        minibatch_stddev: bool = False,
    ) -> None:
        super().__init__()
        if image_size != 64:
            raise ValueError("This compact DCGAN implementation currently supports image_size=64.")
        ndf = feature_maps
        attention_resolutions = attention_resolutions or set()
        final_channels = ndf * 8 + (1 if minibatch_stddev else 0)
        self.net = nn.Sequential(
            maybe_spectral_norm(nn.Conv2d(channels, ndf, 4, 2, 1, bias=False), spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            *maybe_attention(ndf, 32, attention_resolutions, spectral_norm=spectral_norm),
            *discriminator_refinement(ndf, refine_blocks, spectral_norm, use_batchnorm=False),
            maybe_spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), spectral_norm),
            *normalization(ndf * 2, use_batchnorm),
            nn.LeakyReLU(0.2, inplace=True),
            *maybe_attention(ndf * 2, 16, attention_resolutions, spectral_norm=spectral_norm),
            *discriminator_refinement(ndf * 2, refine_blocks, spectral_norm, use_batchnorm=use_batchnorm),
            maybe_spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), spectral_norm),
            *normalization(ndf * 4, use_batchnorm),
            nn.LeakyReLU(0.2, inplace=True),
            *maybe_attention(ndf * 4, 8, attention_resolutions, spectral_norm=spectral_norm),
            *discriminator_refinement(ndf * 4, refine_blocks, spectral_norm, use_batchnorm=use_batchnorm),
            maybe_spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), spectral_norm),
            *normalization(ndf * 8, use_batchnorm),
            nn.LeakyReLU(0.2, inplace=True),
            *discriminator_refinement(ndf * 8, refine_blocks, spectral_norm, use_batchnorm=use_batchnorm),
            *([MinibatchStdDev()] if minibatch_stddev else []),
            maybe_spectral_norm(nn.Conv2d(final_channels, 1, 4, 1, 0, bias=False), spectral_norm),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image).view(-1)


def build_generator_from_config(config: dict) -> Generator:
    architecture = str(config.get("architecture", "dcgan")).lower()
    if architecture in {"residual", "wgan_gp"}:
        return ResidualGenerator(
            latent_dim=int(config.get("latent_dim", 256)),
            channels=int(config.get("channels", 3)),
            feature_maps=int(config.get("generator_features", 64)),
            image_size=int(config.get("image_size", 64)),
            attention_resolutions=parse_resolutions(config.get("generator_attention_resolutions", [])),
        )
    return Generator(
        latent_dim=int(config.get("latent_dim", 100)),
        channels=int(config.get("channels", 3)),
        feature_maps=int(config.get("generator_features", 64)),
        image_size=int(config.get("image_size", 64)),
        refine_blocks=int(config.get("generator_refine_blocks", 0)),
        attention_resolutions=parse_resolutions(config.get("generator_attention_resolutions", [])),
        upsample_mode=str(config.get("generator_upsample_mode", "transpose")),
    )


def build_discriminator_from_config(config: dict) -> Discriminator:
    architecture = str(config.get("architecture", "dcgan")).lower()
    if architecture in {"residual", "wgan_gp"}:
        return ResidualCritic(
            channels=int(config.get("channels", 3)),
            feature_maps=int(config.get("discriminator_features", 64)),
            image_size=int(config.get("image_size", 64)),
            attention_resolutions=parse_resolutions(config.get("discriminator_attention_resolutions", [])),
        )
    return Discriminator(
        channels=int(config.get("channels", 3)),
        feature_maps=int(config.get("discriminator_features", 64)),
        image_size=int(config.get("image_size", 64)),
        spectral_norm=bool(config.get("spectral_norm", False)),
        refine_blocks=int(config.get("discriminator_refine_blocks", 0)),
        attention_resolutions=parse_resolutions(config.get("discriminator_attention_resolutions", [])),
        use_batchnorm=bool(config.get("discriminator_batchnorm", True)),
        minibatch_stddev=bool(config.get("minibatch_stddev", False)),
    )
