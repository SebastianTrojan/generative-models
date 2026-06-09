from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def default_group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def extract(buffer: torch.Tensor, timesteps: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    values = buffer.gather(0, timesteps)
    return values.view(timesteps.size(0), *((1,) * (len(shape) - 1)))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000) / max(1, half - 1)
        frequencies = torch.exp(torch.arange(half, device=timesteps.device) * -scale)
        args = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([args.sin(), args.cos()], dim=1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(default_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(default_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(time_embedding)).view(time_embedding.size(0), -1, 1, 1)
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(default_group_count(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        q = q.view(batch, channels, height * width).transpose(1, 2)
        k = k.view(batch, channels, height * width)
        v = v.view(batch, channels, height * width).transpose(1, 2)
        attention = torch.softmax(torch.bmm(q, k) * (channels**-0.5), dim=-1)
        h = torch.bmm(attention, v).transpose(1, 2).view(batch, channels, height, width)
        return x + self.proj(h)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class DenoisingUNet(nn.Module):
    def __init__(
        self,
        image_size: int = 64,
        channels: int = 3,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (16,),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if image_size != 64:
            raise ValueError("This compact DDPM U-Net currently supports image_size=64.")
        self.image_size = image_size
        self.channels = channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = set(attention_resolutions)

        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_conv = nn.Conv2d(channels, base_channels, 3, 1, 1)
        self.downs = nn.ModuleList()
        skip_channels = [base_channels]
        in_channels = base_channels
        resolution = image_size

        for level, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            blocks = nn.ModuleList()
            attentions = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock(in_channels, out_channels, time_dim, dropout))
                in_channels = out_channels
                attentions.append(AttentionBlock(in_channels) if resolution in self.attention_resolutions else nn.Identity())
                skip_channels.append(in_channels)
            downsample = Downsample(in_channels) if level != len(channel_mults) - 1 else nn.Identity()
            self.downs.append(nn.ModuleDict({"blocks": blocks, "attentions": attentions, "downsample": downsample}))
            if level != len(channel_mults) - 1:
                resolution //= 2
                skip_channels.append(in_channels)

        self.mid_block1 = ResidualBlock(in_channels, in_channels, time_dim, dropout)
        self.mid_attention = AttentionBlock(in_channels)
        self.mid_block2 = ResidualBlock(in_channels, in_channels, time_dim, dropout)

        self.ups = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult
            blocks = nn.ModuleList()
            attentions = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                skip_channel = skip_channels.pop()
                blocks.append(ResidualBlock(in_channels + skip_channel, out_channels, time_dim, dropout))
                in_channels = out_channels
                attentions.append(AttentionBlock(in_channels) if resolution in self.attention_resolutions else nn.Identity())
            upsample = Upsample(in_channels) if level != 0 else nn.Identity()
            self.ups.append(nn.ModuleDict({"blocks": blocks, "attentions": attentions, "upsample": upsample}))
            if level != 0:
                resolution *= 2

        self.output_norm = nn.GroupNorm(default_group_count(in_channels), in_channels)
        self.output_conv = nn.Conv2d(in_channels, channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_mlp(timesteps)
        h = self.input_conv(x)
        skips = [h]

        for down in self.downs:
            for block, attention in zip(down["blocks"], down["attentions"]):
                h = attention(block(h, time_embedding))
                skips.append(h)
            if not isinstance(down["downsample"], nn.Identity):
                h = down["downsample"](h)
                skips.append(h)

        h = self.mid_block1(h, time_embedding)
        h = self.mid_attention(h)
        h = self.mid_block2(h, time_embedding)

        for up in self.ups:
            for block, attention in zip(up["blocks"], up["attentions"]):
                h = torch.cat([h, skips.pop()], dim=1)
                h = attention(block(h, time_embedding))
            if not isinstance(up["upsample"], nn.Identity):
                h = up["upsample"](h)

        return self.output_conv(F.silu(self.output_norm(h)))


def make_beta_schedule(
    schedule: str,
    timesteps: int,
    beta_start: float,
    beta_end: float,
) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
    if schedule == "cosine":
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(0.0001, 0.999)
    raise ValueError(f"Unknown beta schedule: {schedule}")


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        image_size: int = 64,
        channels: int = 3,
        timesteps: int = 500,
        beta_schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps
        betas = make_beta_schedule(beta_schedule, timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape) * noise
        )

    def training_loss(self, model: nn.Module, x_start: torch.Tensor) -> torch.Tensor:
        batch_size = x_start.size(0)
        timesteps = torch.randint(0, self.timesteps, (batch_size,), device=x_start.device)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, timesteps, noise)
        predicted_noise = model(x_noisy, timesteps)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_mean_variance(self, model: nn.Module, x: torch.Tensor, timesteps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        predicted_noise = model(x, timesteps)
        x_start = self.predict_start_from_noise(x, timesteps, predicted_noise).clamp(-1.0, 1.0)
        mean = (
            extract(self.posterior_mean_coef1, timesteps, x.shape) * x_start
            + extract(self.posterior_mean_coef2, timesteps, x.shape) * x
        )
        variance = extract(self.posterior_variance, timesteps, x.shape)
        return mean, variance

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        mean, variance = self.p_mean_variance(model, x, timesteps)
        noise = torch.randn_like(x)
        nonzero_mask = (timesteps != 0).float().view(x.size(0), *((1,) * (x.dim() - 1)))
        return mean + nonzero_mask * torch.sqrt(variance) * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        device: torch.device,
        progress: bool = False,
    ) -> torch.Tensor:
        image = torch.randn(shape, device=device)
        iterator = reversed(range(self.timesteps))
        if progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=self.timesteps, desc="DDPM sampling", leave=False)
        for step in iterator:
            timesteps = torch.full((shape[0],), step, device=device, dtype=torch.long)
            image = self.p_sample(model, image, timesteps)
        return image.clamp(-1.0, 1.0)

    @torch.no_grad()
    def ddim_sample_loop(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        device: torch.device,
        initial_noise: torch.Tensor | None = None,
        progress: bool = False,
    ) -> torch.Tensor:
        image = initial_noise.to(device) if initial_noise is not None else torch.randn(shape, device=device)
        if tuple(image.shape) != shape:
            raise ValueError(f"initial_noise shape {tuple(image.shape)} does not match requested shape {shape}")
        iterator = reversed(range(self.timesteps))
        if progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=self.timesteps, desc="DDIM sampling", leave=False)
        for step in iterator:
            timesteps = torch.full((shape[0],), step, device=device, dtype=torch.long)
            predicted_noise = model(image, timesteps)
            x_start = self.predict_start_from_noise(image, timesteps, predicted_noise).clamp(-1.0, 1.0)
            if step == 0:
                image = x_start
            else:
                alpha_prev = self.alphas_cumprod[step - 1]
                image = torch.sqrt(alpha_prev) * x_start + torch.sqrt(1.0 - alpha_prev) * predicted_noise
        return image.clamp(-1.0, 1.0)


def build_denoiser_from_config(config: dict) -> DenoisingUNet:
    return DenoisingUNet(
        image_size=int(config.get("image_size", 64)),
        channels=int(config.get("channels", 3)),
        base_channels=int(config.get("base_channels", 64)),
        channel_mults=tuple(int(value) for value in config.get("channel_mults", [1, 2, 4, 4])),
        num_res_blocks=int(config.get("num_res_blocks", 2)),
        attention_resolutions=tuple(int(value) for value in config.get("attention_resolutions", [16])),
        dropout=float(config.get("dropout", 0.0)),
    )


def build_diffusion_from_config(config: dict, device: torch.device | None = None) -> GaussianDiffusion:
    diffusion = GaussianDiffusion(
        image_size=int(config.get("image_size", 64)),
        channels=int(config.get("channels", 3)),
        timesteps=int(config.get("timesteps", 500)),
        beta_schedule=str(config.get("beta_schedule", "linear")),
        beta_start=float(config.get("beta_start", 0.0001)),
        beta_end=float(config.get("beta_end", 0.02)),
    )
    if device is not None:
        diffusion = diffusion.to(device)
    return diffusion
