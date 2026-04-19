from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class KernelSpec:
    kind: str
    size: int
    params: Dict[str, float]


def _validate_size(size: int) -> None:
    if size <= 0 or size % 2 == 0:
        raise ValueError("Kernel size must be a positive odd integer.")


def disk_blur_kernel(size: int, radius: float | None = None) -> np.ndarray:
    _validate_size(size)
    if radius is None:
        radius = size / 2.0

    center = size // 2
    y, x = np.ogrid[:size, :size]
    dist2 = (x - center) ** 2 + (y - center) ** 2
    mask = dist2 <= (radius ** 2)

    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[mask] = 1.0
    s = kernel.sum()
    if s == 0:
        raise ValueError("Disk blur kernel is empty; adjust radius/size.")
    return kernel / s


def gaussian_kernel(size: int, sigma: float | None = None) -> np.ndarray:
    _validate_size(size)
    if sigma is None:
        sigma = size / 6.0
    if sigma <= 0:
        raise ValueError("Sigma must be positive.")

    center = size // 2
    y, x = np.ogrid[:size, :size]
    dist2 = (x - center) ** 2 + (y - center) ** 2
    kernel = np.exp(-dist2 / (2 * sigma ** 2)).astype(np.float32)
    return kernel / kernel.sum()


def diagonal_motion_blur_kernel(size: int, thickness: int = 1) -> np.ndarray:
    _validate_size(size)
    if thickness <= 0:
        raise ValueError("Thickness must be positive.")

    kernel = np.zeros((size, size), dtype=np.float32)
    half_t = thickness // 2

    for i in range(size):
        for t in range(-half_t, half_t + 1):
            j = i + t
            if 0 <= j < size:
                kernel[i, j] = 1.0

    s = kernel.sum()
    if s == 0:
        raise ValueError("Diagonal motion kernel is empty.")
    return kernel / s


def numerical_rank(kernel: np.ndarray, tol: float = 1e-5) -> int:
    s = np.linalg.svd(kernel.astype(np.float64), compute_uv=False)
    return int(np.sum(s > tol))


def is_nonseparable(kernel: np.ndarray, tol: float = 1e-5) -> bool:
    return numerical_rank(kernel, tol=tol) > 1
