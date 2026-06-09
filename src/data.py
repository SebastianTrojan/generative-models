from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

from PIL import Image, ImageOps, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .utils import resolve_path


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class CenterSquareCrop:
    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        return image.crop((left, top, left + side, top + side))


def list_image_files(root: str | Path) -> list[Path]:
    root = Path(root)
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def infer_catdog_label(path: str | Path) -> int | None:
    name = Path(path).name.lower()
    if "cat" in name:
        return 0
    if "dog" in name:
        return 1
    return None


def build_transform(image_size: int, train: bool = True) -> Callable:
    steps: list[Callable] = [
        transforms.Lambda(lambda image: ImageOps.exif_transpose(image).convert("RGB")),
        CenterSquareCrop(),
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
    ]
    if train:
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transforms.Compose(steps)


class ImageFolderFlatDataset(Dataset):
    """Recursively load images without requiring class subdirectories."""

    def __init__(
        self,
        root: str | Path,
        image_size: int = 64,
        train: bool = True,
        transform: Callable | None = None,
        return_labels: bool = False,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")
        self.paths = list_image_files(self.root)
        if not self.paths:
            extensions = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise RuntimeError(f"No image files with extensions {extensions} found under {self.root}")
        self.transform = transform or build_transform(image_size=image_size, train=train)
        self.return_labels = return_labels
        self._bad_paths: set[Path] = set()

    def __len__(self) -> int:
        return len(self.paths)

    def _load_image(self, path: Path):
        with Image.open(path) as image:
            return self.transform(image)

    def __getitem__(self, index: int):
        start_index = index % len(self.paths)
        for offset in range(len(self.paths)):
            current_index = (start_index + offset) % len(self.paths)
            path = self.paths[current_index]
            if path in self._bad_paths:
                continue
            try:
                image = self._load_image(path)
                if self.return_labels:
                    label = infer_catdog_label(path)
                    return image, -1 if label is None else label
                return image
            except (OSError, UnidentifiedImageError, ValueError):
                self._bad_paths.add(path)
                continue

        random_path = random.choice(self.paths)
        raise RuntimeError(f"All image loading attempts failed; last sampled path was {random_path}")


def build_dataloader(config: dict, train: bool = True, return_labels: bool = False) -> tuple[ImageFolderFlatDataset, DataLoader]:
    dataset_root = resolve_path(config["dataset_root"], config.get("_config_path"), must_exist=True)
    dataset = ImageFolderFlatDataset(
        root=dataset_root,
        image_size=int(config.get("image_size", 64)),
        train=train,
        return_labels=return_labels,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 128)),
        shuffle=train,
        num_workers=int(config.get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
    )
    return dataset, loader


def preprocess_pil_image(image: Image.Image, image_size: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = CenterSquareCrop()(image)
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    return image


def export_preprocessed_images(
    dataset_root: str | Path,
    out_dir: str | Path,
    image_size: int,
    limit: int | None = None,
) -> int:
    dataset_root = Path(dataset_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob("real_*.png"))
    if limit is not None and len(existing) >= limit:
        return limit

    count = 0
    for path in list_image_files(dataset_root):
        if limit is not None and count >= limit:
            break
        out_path = out_dir / f"real_{count:06d}.png"
        if out_path.exists():
            count += 1
            continue
        try:
            with Image.open(path) as image:
                preprocessed = preprocess_pil_image(image, image_size)
            preprocessed.save(out_path)
            count += 1
        except (OSError, UnidentifiedImageError, ValueError):
            continue
    return count
