from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATE_KERNELS_PATH = REPO_ROOT / "person2" / "scripts" / "generate_kernels.py"

FILTER_CHOICES = [
    ("gaussian", "Gaussian Blur"),
    ("disk", "Disk Blur"),
    ("motion_diag", "Diagonal Motion Blur"),
]

FILTER_LABELS = dict(FILTER_CHOICES)


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _kernel_module():
    return _load_module("person2_generate_kernels", GENERATE_KERNELS_PATH)


def build_kernel(
    filter_type: str,
    kernel_size: int,
    sigma: float | None = None,
    disk_radius: float | None = None,
    motion_thickness: int = 1,
) -> np.ndarray:
    module = _kernel_module()
    if filter_type == "gaussian":
        actual_sigma = sigma if sigma and sigma > 0 else kernel_size / 6.0
        return np.asarray(module.gaussian_kernel(kernel_size, actual_sigma), dtype=np.float32)
    if filter_type == "disk":
        actual_radius = disk_radius if disk_radius and disk_radius > 0 else None
        return np.asarray(module.disk_kernel(kernel_size, radius=actual_radius), dtype=np.float32)
    if filter_type == "motion_diag":
        return np.asarray(module.motion_diag_kernel(kernel_size, thickness=motion_thickness), dtype=np.float32)
    raise ValueError(f"Unsupported filter type: {filter_type}")


def kernel_parameter_summary(
    filter_type: str,
    kernel_size: int,
    sigma: float | None = None,
    disk_radius: float | None = None,
    motion_thickness: int = 1,
) -> dict[str, float | int | str]:
    summary: dict[str, float | int | str] = {
        "filter_type": filter_type,
        "filter_label": FILTER_LABELS[filter_type],
        "kernel_size": int(kernel_size),
    }
    if filter_type == "gaussian":
        summary["sigma"] = float(sigma if sigma and sigma > 0 else kernel_size / 6.0)
    elif filter_type == "disk":
        summary["disk_radius"] = float(disk_radius if disk_radius and disk_radius > 0 else kernel_size / 3.0)
    elif filter_type == "motion_diag":
        summary["motion_thickness"] = int(motion_thickness)
    return summary
