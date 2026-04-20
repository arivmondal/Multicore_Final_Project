from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def to_float01(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.max() > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def compute_image_metrics(reference: np.ndarray, test: np.ndarray) -> dict[str, float]:
    ref = to_float01(reference)
    tst = to_float01(test)
    diff = ref - tst
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    if mse == 0.0:
        psnr = float("inf")
    else:
        psnr = float(10.0 * np.log10(1.0 / mse))
    if ref.ndim == 2:
        ssim = float(structural_similarity(ref, tst, data_range=1.0))
    else:
        ssim = float(structural_similarity(ref, tst, channel_axis=-1, data_range=1.0))
    return {
        "mse": mse,
        "mae": mae,
        "psnr": psnr,
        "ssim": ssim,
        "max_abs_difference": max_abs,
    }


def compute_kernel_rank_metrics(kernel: np.ndarray, rank: int) -> dict[str, object]:
    kernel64 = np.asarray(kernel, dtype=np.float64)
    u, singular_values, vt = np.linalg.svd(kernel64, full_matrices=True)
    reconstruction = (u[:, :rank] * singular_values[:rank]) @ vt[:rank, :]
    diff = kernel64 - reconstruction
    fro_error = float(np.linalg.norm(diff, ord="fro"))
    fro_norm = float(np.linalg.norm(kernel64, ord="fro"))
    cumulative_energy = float(
        np.sum(np.square(singular_values[:rank])) / np.sum(np.square(singular_values))
    )
    return {
        "singular_values": singular_values.astype(np.float64),
        "reconstruction": reconstruction.astype(np.float32),
        "frobenius_error": fro_error,
        "relative_frobenius_error": 0.0 if fro_norm == 0.0 else float(fro_error / fro_norm),
        "cumulative_energy_retained": cumulative_energy,
    }
