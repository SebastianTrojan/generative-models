# Project III: Generative Models for Cat Image Synthesis

This project implements a realistic one-week PyTorch pipeline for 64x64 image generation:

- DCGAN / improved DCGAN as the main model.
- Improved residual convolutional VAE as a baseline.
- Optional compact DDPM as a compute-heavier diffusion experiment.
- Optional exploratory DCGAN training on the combined cats-and-dogs dataset.
- FID, sample grids, diversity checks, and latent interpolation artifacts for the report.

The default settings target a consumer GPU and prioritize finishing a complete, reproducible experiment. DDPM support is included as an optional extension, but it is much slower to train and sample than the GAN/VAE models.

## Setup

Use Python 3.11+.

```bash
cd generative-models
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The configs in `configs/` already point to the datasets present in this workspace:

```text
data/raw/cats
data/raw/catdog
```

If you move the datasets, update `dataset_root` in the YAML files. The loader recursively finds `.jpg`, `.jpeg`, `.png`, and `.webp` images, converts them to RGB, center-crops to a square, resizes to 64x64, optionally flips during training, and normalizes tensors to `[-1, 1]`.

## Train Models

Train the main cat DCGAN:

```bash
python -m src.train_dcgan --config configs/dcgan_cat64.yaml
```

Train a heavier improved DCGAN variant:

```bash
python -m src.train_dcgan --config configs/dcgan_cat64_deep.yaml
```

Train the strongest DCGAN configuration:

```bash
python -m src.train_dcgan --config configs/dcgan_cat64_best.yaml
```

This configuration uses a residual convolutional generator and critic trained with WGAN-GP, multiple critic updates per generator update, and EMA generator checkpoints. It is slower than the baseline DCGAN but is designed to avoid the zero-logit failure mode that can happen with brittle hinge/spectral setups.

Train the improved residual VAE baseline:

```bash
python -m src.train_vae --config configs/vae_cat64.yaml
```

This VAE is not compatible with checkpoints from the older simple VAE architecture. Start it from scratch; it saves under `outputs/checkpoints/vae_cat64_best/`.

Train the optional compact DDPM:

```bash
python -m src.train_ddpm --config configs/ddpm_cat64.yaml
```

Run the exploratory cats+dogs DCGAN:

```bash
python -m src.train_dcgan --config configs/dcgan_catsdogs64.yaml
```

Checkpoints are saved under `outputs/checkpoints/<run_name>/`, and sample grids under `outputs/samples/<run_name>/`.

Resume DCGAN training from a training checkpoint:

```bash
python -m src.train_dcgan --config configs/dcgan_cat64.yaml --resume outputs/checkpoints/dcgan_cat64/training_latest.pt
```

Resume VAE training:

```bash
python -m src.train_vae --config configs/vae_cat64.yaml --resume outputs/checkpoints/vae_cat64_best/training_latest.pt
```

## Generate Images

DCGAN:

```bash
python -m src.generate \
  --model dcgan \
  --checkpoint outputs/checkpoints/dcgan_cat64/generator_latest.pt \
  --num-images 100 \
  --out-dir outputs/generated/dcgan_cat64
```

Best DCGAN, using the EMA generator:

```bash
python -m src.generate \
  --model dcgan \
  --checkpoint outputs/checkpoints/dcgan_cat64_wgangp_best/generator_ema_latest.pt \
  --config configs/dcgan_cat64_best.yaml \
  --num-images 100 \
  --out-dir outputs/generated/dcgan_cat64_wgangp_best
```

VAE:

```bash
python -m src.generate \
  --model vae \
  --checkpoint outputs/checkpoints/vae_cat64_best/model_latest.pt \
  --num-images 100 \
  --out-dir outputs/generated/vae_cat64_best
```

DDPM:

```bash
python -m src.generate \
  --model ddpm \
  --checkpoint outputs/checkpoints/ddpm_cat64/model_latest.pt \
  --config configs/ddpm_cat64.yaml \
  --num-images 16 \
  --batch-size 8 \
  --out-dir outputs/generated/ddpm_cat64
```

Each command saves individual PNG files and a grid image.

## Latent Interpolation

For DCGAN latent-noise interpolation:

```bash
python -m src.interpolate \
  --model dcgan \
  --checkpoint outputs/checkpoints/dcgan_cat64/generator_latest.pt \
  --config configs/dcgan_cat64.yaml
```

For the best DCGAN, use:

```bash
python -m src.interpolate \
  --model dcgan \
  --checkpoint outputs/checkpoints/dcgan_cat64_wgangp_best/generator_ema_latest.pt \
  --config configs/dcgan_cat64_best.yaml
```

For VAE latent interpolation:

```bash
python -m src.interpolate \
  --model vae \
  --checkpoint outputs/checkpoints/vae_cat64_best/model_latest.pt \
  --config configs/vae_cat64.yaml
```

For DDPM initial-noise interpolation:

```bash
python -m src.interpolate \
  --model ddpm \
  --checkpoint outputs/checkpoints/ddpm_cat64/model_latest.pt \
  --config configs/ddpm_cat64.yaml
```

This samples two latent noise tensors, linearly interpolates between them with 10 evenly spaced points, and saves:

```text
outputs/interpolations/<run_name>/interpolation_grid.png
outputs/interpolations/<run_name>/latent_vectors.npz
```

The `.npz` file contains `z_a`, `z_b`, all 10 interpolated latent/noise matrices, and interpolation coefficients. For DDPM, the interpolated matrices are image-shaped initial noise tensors and sampling uses a deterministic DDIM-style reverse process. In the report, smooth transitions suggest the model learned a meaningful generation space; abrupt jumps or near-identical images can indicate poor coverage or mode collapse.

## FID

Compute FID for DCGAN:

```bash
python -m src.evaluate_fid \
  --model dcgan \
  --checkpoint outputs/checkpoints/dcgan_cat64/generator_latest.pt \
  --config configs/dcgan_cat64.yaml \
  --num-images 5000
```

Compute FID for the best DCGAN:

```bash
python -m src.evaluate_fid \
  --model dcgan \
  --checkpoint outputs/checkpoints/dcgan_cat64_wgangp_best/generator_ema_latest.pt \
  --config configs/dcgan_cat64_best.yaml \
  --num-images 5000
```

Compute FID for VAE:

```bash
python -m src.evaluate_fid \
  --model vae \
  --checkpoint outputs/checkpoints/vae_cat64_best/model_latest.pt \
  --config configs/vae_cat64.yaml \
  --num-images 5000
```

Compute FID for DDPM. This is slow, so start with a small number:

```bash
python -m src.evaluate_fid \
  --model ddpm \
  --checkpoint outputs/checkpoints/ddpm_cat64/model_latest.pt \
  --config configs/ddpm_cat64.yaml \
  --num-images 500 \
  --batch-size 8
```

For quick smoke tests, use `--num-images 1000` or smaller. Results are written to `outputs/fid/<run_name>_fid.json`. The script uses `clean-fid` first and falls back to `pytorch-fid` if needed.

## Diversity And Mode Collapse

Run a practical diversity check:

```bash
python -m src.diversity \
  --model dcgan \
  --checkpoint outputs/checkpoints/dcgan_cat64/generator_latest.pt \
  --config configs/dcgan_cat64.yaml \
  --num-images 256
```

For the best DCGAN:

```bash
python -m src.diversity \
  --model dcgan \
  --checkpoint outputs/checkpoints/dcgan_cat64_wgangp_best/generator_ema_latest.pt \
  --config configs/dcgan_cat64_best.yaml \
  --num-images 256
```

For VAE:

```bash
python -m src.diversity \
  --model vae \
  --checkpoint outputs/checkpoints/vae_cat64_best/model_latest.pt \
  --config configs/vae_cat64.yaml \
  --num-images 256
```

For DDPM:

```bash
python -m src.diversity \
  --model ddpm \
  --checkpoint outputs/checkpoints/ddpm_cat64/model_latest.pt \
  --config configs/ddpm_cat64.yaml \
  --num-images 64 \
  --batch-size 8
```

This saves a generated sample grid and JSON metrics such as average pairwise pixel distance. These numbers are not a replacement for visual inspection or FID, but they are useful for discussing mode collapse.

If many samples look identical:

- Lower the discriminator learning rate or train it less aggressively.
- Use real-label smoothing, e.g. `real_label_smoothing: 0.9`.
- Add small instance noise, e.g. `instance_noise_std: 0.05`.
- Try `spectral_norm: true`.
- Reduce the learning rate from `0.0002` to `0.0001`.
- Inspect whether the dataset preprocessing is too narrow or too repetitive.

## Suggested One-Week Experiment Plan

1. Train `dcgan_cat64` for a shorter smoke run, then for 100 epochs.
2. Save qualitative grids every 5 epochs and identify the best-looking checkpoint.
3. Train hyperparameter variants by copying `configs/dcgan_cat64.yaml`:
   - learning rate `0.0002` vs `0.0001`,
   - latent dimension `100` vs `256`,
   - label smoothing on/off,
   - spectral normalization on/off if mode collapse appears.
4. Train `dcgan_cat64_deep` if the baseline trains quickly or underfits. This version uses wider layers, latent dimension 256, one refinement conv block per scale, spectral normalization, and light instance noise.
5. Train `dcgan_cat64_wgangp_best` as the final improved GAN candidate and compare it against the baseline DCGAN.
6. Train `vae_cat64_best` as the improved residual VAE baseline.
7. Optionally train `ddpm_cat64`. This is slower because each generated sample requires hundreds of denoising steps.
8. Compute FID for DCGAN, best DCGAN, VAE, and optionally DDPM using the same number of generated images when runtime allows.
9. Run `src.interpolate` for DCGAN, VAE, and DDPM.
10. Run diversity checks for DCGAN, VAE, and DDPM.
11. Run `dcgan_catsdogs64` for 50 epochs as an exploratory extension and compare whether samples look class-distinct or blended.

## Report Notes

Use both quantitative and qualitative evidence. FID compares generated and real-image feature distributions; lower is usually better, but it can disagree with human judgment. VAE samples often have smoother, blurrier textures because the reconstruction objective averages plausible outputs. DCGAN samples may be sharper but can suffer from unstable training or mode collapse.

For mixed cats+dogs training, an unconditional GAN has no label signal. It may generate cats, dogs, or ambiguous animal faces. Class-distinct generation usually requires conditioning, labels, or a stronger model, so treat this part as exploratory rather than the central result.
