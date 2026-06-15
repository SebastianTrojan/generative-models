# Project III: Generative Models for Cat Image Synthesis

This project implements a realistic one-week PyTorch pipeline for 64x64 image generation:

- DCGAN as the main GAN model.
- Improved residual convolutional VAE as a baseline.
- Optional compact DDPM as a compute-heavier diffusion experiment.
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

Train the cat DCGAN:

```bash
python -m src.train_dcgan --config configs/dcgan_cat64_baseline.yaml
```

This configuration uses a DCGAN generator/discriminator trained with LSGAN loss, conservative discriminator learning rate, gradient clipping, and finite-loss guards. It saves under `outputs/checkpoints/dcgan_cat64_baseline/`.

Run the DCGAN hyperparameter sweep:

```bash
PYTHON_BIN=./venv/bin/python bash scripts/run_dcgan_experiments.sh --device cuda
```

For a quick comparison run, override epochs:

```bash
PYTHON_BIN=./venv/bin/python bash scripts/run_dcgan_experiments.sh --epochs 30 --device cuda
```

Train the improved residual VAE baseline:

```bash
python -m src.train_vae --config configs/vae_cat64_baseline.yaml
```

This VAE is not compatible with older VAE checkpoints. Start it from scratch; it saves under `outputs/checkpoints/vae_cat64_baseline/`.

Run the VAE hyperparameter sweep:

```bash
PYTHON_BIN=./venv/bin/python bash scripts/run_vae_experiments.sh --device cuda
```

For a quick comparison run, override epochs:

```bash
PYTHON_BIN=./venv/bin/python bash scripts/run_vae_experiments.sh --epochs 30 --device cuda
```

Train the optional compact DDPM:

```bash
python -m src.train_ddpm --config configs/ddpm_cat64_baseline.yaml
```

Checkpoints are saved under `outputs/checkpoints/<run_name>/`, and sample grids under `outputs/samples/<run_name>/`.

Resume DCGAN training from a training checkpoint:

```bash
python -m src.train_dcgan --config configs/dcgan_cat64_baseline.yaml --resume outputs/checkpoints/dcgan_cat64_baseline/training_latest.pt
```

Resume VAE training:

```bash
python -m src.train_vae --config configs/vae_cat64_baseline.yaml --resume outputs/checkpoints/vae_cat64_baseline/training_latest.pt
```

## Generate Images

DCGAN:

```bash
python -m src.generate \
  --model dcgan \
  --checkpoint outputs/checkpoints/dcgan_cat64_baseline/generator_latest.pt \
  --config configs/dcgan_cat64_baseline.yaml \
  --num-images 100 \
  --out-dir outputs/generated/dcgan_cat64_baseline
```

VAE:

```bash
python -m src.generate \
  --model vae \
  --checkpoint outputs/checkpoints/vae_cat64_baseline/model_latest.pt \
  --config configs/vae_cat64_baseline.yaml \
  --num-images 100 \
  --out-dir outputs/generated/vae_cat64_baseline
```

DDPM:

```bash
python -m src.generate \
  --model ddpm \
  --checkpoint outputs/checkpoints/ddpm_cat64_baseline/model_latest.pt \
  --config configs/ddpm_cat64_baseline.yaml \
  --num-images 16 \
  --batch-size 8 \
  --out-dir outputs/generated/ddpm_cat64_baseline
```

Each command saves individual PNG files and a grid image.

## Latent Interpolation

For DCGAN latent-noise interpolation:

```bash
python -m src.interpolate \
  --model dcgan \
  --checkpoint outputs/checkpoints/dcgan_cat64_baseline/generator_latest.pt \
  --config configs/dcgan_cat64_baseline.yaml
```

For VAE latent interpolation:

```bash
python -m src.interpolate \
  --model vae \
  --checkpoint outputs/checkpoints/vae_cat64_baseline/model_latest.pt \
  --config configs/vae_cat64_baseline.yaml
```

For DDPM initial-noise interpolation:

```bash
python -m src.interpolate \
  --model ddpm \
  --checkpoint outputs/checkpoints/ddpm_cat64_baseline/model_latest.pt \
  --config configs/ddpm_cat64_baseline.yaml
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
  --checkpoint outputs/checkpoints/dcgan_cat64_baseline/generator_latest.pt \
  --config configs/dcgan_cat64_baseline.yaml \
  --num-images 5000
```

Compute FID for VAE:

```bash
python -m src.evaluate_fid \
  --model vae \
  --checkpoint outputs/checkpoints/vae_cat64_baseline/model_latest.pt \
  --config configs/vae_cat64_baseline.yaml \
  --num-images 5000
```

Compute FID for DDPM. This is slow, so start with a small number:

```bash
python -m src.evaluate_fid \
  --model ddpm \
  --checkpoint outputs/checkpoints/ddpm_cat64_baseline/model_latest.pt \
  --config configs/ddpm_cat64_baseline.yaml \
  --num-images 500 \
  --batch-size 8
```

For quick smoke tests, use `--num-images 1000` or smaller. Results are written to `outputs/fid/<run_name>_fid.json`. The script uses `clean-fid` first and falls back to `pytorch-fid` if needed.

## Diversity And Mode Collapse

Run a practical diversity check:

```bash
python -m src.diversity \
  --model dcgan \
  --checkpoint outputs/checkpoints/dcgan_cat64_baseline/generator_latest.pt \
  --config configs/dcgan_cat64_baseline.yaml \
  --num-images 256
```

For VAE:

```bash
python -m src.diversity \
  --model vae \
  --checkpoint outputs/checkpoints/vae_cat64_baseline/model_latest.pt \
  --config configs/vae_cat64_baseline.yaml \
  --num-images 256
```

For DDPM:

```bash
python -m src.diversity \
  --model ddpm \
  --checkpoint outputs/checkpoints/ddpm_cat64_baseline/model_latest.pt \
  --config configs/ddpm_cat64_baseline.yaml \
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

1. Train `dcgan_cat64_baseline` with `configs/dcgan_cat64_baseline.yaml`.
2. Save qualitative grids every 5 epochs and identify the best-looking checkpoint.
3. Train `vae_cat64_baseline` as the improved residual VAE baseline.
4. Optionally train DDPM variants. DDPM is slower because each generated sample requires hundreds of denoising steps.
5. Compute FID for DCGAN, VAE, and DDPM using the same number of generated images when runtime allows.
6. Run `src.interpolate` for DCGAN, VAE, and DDPM.
7. Run diversity checks for DCGAN, VAE, and DDPM.

## Report Notes

Use both quantitative and qualitative evidence. FID compares generated and real-image feature distributions; lower is usually better, but it can disagree with human judgment. VAE samples often have smoother, blurrier textures because the reconstruction objective averages plausible outputs. DCGAN samples may be sharper but can suffer from training instability or mode collapse.

For mixed cats+dogs training, an unconditional GAN has no label signal. If you run that extension, copy the DCGAN config and change only `run_name` and `dataset_root`. Class-distinct generation usually requires conditioning, labels, or a stronger model, so treat this part as exploratory rather than the central result.
