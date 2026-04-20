from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

_VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def ensure_output_dirs(base_output: Path) -> tuple[Path, Path]:
    out_images = base_output / "images"
    out_reports = base_output / "reports"
    out_images.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)
    return out_images, out_reports


def _read_image(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 255.0).astype(np.float32)


def load_images_or_synthetic(input_images_dir: Path) -> List[Tuple[str, np.ndarray]]:
    input_images_dir.mkdir(parents=True, exist_ok=True)
    found = sorted([p for p in input_images_dir.iterdir() if p.suffix.lower() in _VALID_EXTS])
    if found:
        return [(p.stem, _read_image(p)) for p in found]

    rng = np.random.default_rng(42)
    out: List[Tuple[str, np.ndarray]] = []

    for size in (512, 1024, 2048):
        y, x = np.mgrid[0:size, 0:size]
        gradient = np.stack(
            [x / float(size - 1), y / float(size - 1), 0.5 * np.ones_like(x, dtype=np.float32)],
            axis=-1,
        ).astype(np.float32)

        checker = (((x // 32 + y // 32) % 2).astype(np.float32))
        checker_rgb = np.stack([checker, 1.0 - checker, 0.5 * checker], axis=-1).astype(np.float32)

        noise = rng.random((size, size, 3), dtype=np.float32)
        out.append((f"synthetic_gradient_{size}", gradient))
        out.append((f"synthetic_checker_{size}", checker_rgb))
        out.append((f"synthetic_noise_{size}", noise))

    return out


def save_rgb01_image(path: Path, image: np.ndarray) -> None:
    arr = np.clip(image, 0.0, 1.0)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    bgr = cv2.cvtColor((arr * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr)
