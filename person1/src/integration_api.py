from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .convolution_runner import CudaExactConvolver
from .io_utils import save_rgb01_image
from .kernels import diagonal_motion_blur_kernel, disk_blur_kernel
from .metrics import compute_error_stats, compute_psnr, compute_ssim


def _read_image_rgb01(image_path: Path) -> np.ndarray:
    """Read an image from disk and convert it to RGB float32 in [0, 1]."""
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 255.0).astype(np.float32)


def _build_kernel(kernel_type: str, kernel_size: int, kernel_params: dict[str, Any] | None = None) -> np.ndarray:
    """Create a nonseparable kernel by name and size."""
    params = kernel_params or {}
    if kernel_type == "disk":
        return disk_blur_kernel(kernel_size, radius=params.get("radius"))
    if kernel_type == "diag_motion":
        return diagonal_motion_blur_kernel(kernel_size, thickness=int(params.get("thickness", 1)))
    raise ValueError("Unsupported kernel_type. Use 'disk' or 'diag_motion'.")

# ========== PUBLIC API ==========

# Example usage:
# from src import run_exact_cuda_from_paths, compute_metrics_from_paths

# run_info = run_exact_cuda_from_paths(
#     input_image_path="task1/input/images/sample_gradient.png",
#     output_image_path="task1/output/images/sample_gradient_disk31.png",
#     kernel_type="disk",
#     kernel_size=31,
#     kernel_params={"radius": 15.5},
# )

# metrics = compute_metrics_from_paths(
#     reference_image_path="task1/output/images/reference.png",
#     test_image_path="task1/output/images/sample_gradient_disk31.png",
# )
# print(run_info)
# print(metrics)

def run_exact_cuda_from_paths(
    input_image_path: str | Path,
    output_image_path: str | Path,
    kernel_type: str,
    kernel_size: int,
    kernel_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run exact nonseparable 2-D CUDA convolution from file paths.

    Inputs:
    - input_image_path: Path to source image file.
    - output_image_path: Path where filtered output image will be saved.
    - kernel_type: Kernel name ('disk' or 'diag_motion').
    - kernel_size: Odd square kernel size.
    - kernel_params: Optional kernel parameters.
      - disk: {'radius': float}
      - diag_motion: {'thickness': int}

    Outputs:
    - Returns a dictionary with run metadata:
      - input_image_path, output_image_path
      - kernel_type, kernel_size, kernel_params
      - image_h, image_w, channels
    - Writes the filtered image to output_image_path.
    """
    in_path = Path(input_image_path)
    out_path = Path(output_image_path)

    image = _read_image_rgb01(in_path)
    kernel = _build_kernel(kernel_type=kernel_type, kernel_size=kernel_size, kernel_params=kernel_params)

    convolver = CudaExactConvolver(border_mode="replicate")
    filtered = convolver.convolve(image, kernel)
    save_rgb01_image(out_path, filtered)

    channels = 1 if filtered.ndim == 2 else int(filtered.shape[2])
    return {
        "input_image_path": str(in_path),
        "output_image_path": str(out_path),
        "kernel_type": kernel_type,
        "kernel_size": int(kernel_size),
        "kernel_params": kernel_params or {},
        "image_h": int(filtered.shape[0]),
        "image_w": int(filtered.shape[1]),
        "channels": channels,
    }


def compute_metrics_from_paths(reference_image_path: str | Path, test_image_path: str | Path) -> dict[str, float]:
    """
    Compute PSNR/SSIM and error stats from two image file paths.

    Inputs:
    - reference_image_path: Path to the reference image.
    - test_image_path: Path to the image being evaluated.

    Outputs:
    - Returns a dictionary with:
      - psnr: Peak Signal-to-Noise Ratio.
      - ssim: Structural Similarity Index.
      - mae: Mean Absolute Error.
      - max_abs_error: Maximum absolute pixel error.
    """
    ref = _read_image_rgb01(Path(reference_image_path))
    tst = _read_image_rgb01(Path(test_image_path))

    if ref.shape != tst.shape:
        raise ValueError(f"Image shape mismatch: reference {ref.shape} vs test {tst.shape}")

    errs = compute_error_stats(ref, tst)
    return {
        "psnr": compute_psnr(ref, tst),
        "ssim": compute_ssim(ref, tst),
        "mae": errs["mae"],
        "max_abs_error": errs["max_abs_error"],
    }
