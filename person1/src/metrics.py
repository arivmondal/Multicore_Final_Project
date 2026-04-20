from __future__ import annotations

from typing import Dict

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def _to_float01(img: np.ndarray) -> np.ndarray:
    arr = img.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def compute_psnr(reference: np.ndarray, test: np.ndarray) -> float:
    ref = _to_float01(reference)
    tst = _to_float01(test)
    return float(cv2.PSNR(ref, tst, 1.0))


def compute_ssim(reference: np.ndarray, test: np.ndarray) -> float:
    ref = _to_float01(reference)
    tst = _to_float01(test)

    if ref.ndim == 2:
        score = structural_similarity(ref, tst, data_range=1.0)
    else:
        score = structural_similarity(ref, tst, channel_axis=-1, data_range=1.0)
    return float(score)


def compute_error_stats(reference: np.ndarray, test: np.ndarray) -> Dict[str, float]:
    ref = _to_float01(reference)
    tst = _to_float01(test)
    diff = np.abs(ref - tst)
    return {
        "mae": float(np.mean(diff)),
        "max_abs_error": float(np.max(diff)),
    }
