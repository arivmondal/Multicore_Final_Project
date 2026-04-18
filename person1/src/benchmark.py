from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Iterable, List

import numpy as np

from .convolution_runner import CudaExactConvolver
from .io_utils import save_rgb01_image
from .kernels import diagonal_motion_blur_kernel, disk_blur_kernel


@dataclass
class BenchmarkRow:
    timestamp_utc: str
    image_id: str
    image_h: int
    image_w: int
    channels: int
    kernel_type: str
    kernel_size: int
    run_count: int
    warmup_count: int
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    throughput_mpix_s: float
    rank: int
    approximation_method: str
    reconstruction_error: float


def _resize_to(image: np.ndarray, size: int) -> np.ndarray:
    if image.shape[0] == size and image.shape[1] == size:
        return image
    import cv2

    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32)


def _kernel_specs(kernel_sizes: Iterable[int]):
    for k in kernel_sizes:
        yield "disk", disk_blur_kernel(k)
        yield "diag_motion", diagonal_motion_blur_kernel(k)


def run_benchmark(
    convolver: CudaExactConvolver,
    images: list[tuple[str, np.ndarray]],
    image_sizes: Iterable[int],
    kernel_sizes: Iterable[int],
    warmup: int,
    runs: int,
    output_images_dir: Path,
) -> List[BenchmarkRow]:
    rows: List[BenchmarkRow] = []

    for image_id, image in images:
        for s in image_sizes:
            img = _resize_to(image, s)
            channels = 1 if img.ndim == 2 else img.shape[2]

            for kname, kernel in _kernel_specs(kernel_sizes):
                for _ in range(warmup):
                    _ = convolver.benchmark_once_ms(img, kernel)

                timings = [convolver.benchmark_once_ms(img, kernel) for _ in range(runs)]
                gpu_out = convolver.convolve(img, kernel)

                out_name = f"{image_id}_{s}_{kname}_k{kernel.shape[0]}.png"
                save_rgb01_image(output_images_dir / out_name, gpu_out)

                mpix = (img.shape[0] * img.shape[1]) / 1_000_000.0
                mean_ms = float(mean(timings))
                row = BenchmarkRow(
                    timestamp_utc=datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    image_id=image_id,
                    image_h=img.shape[0],
                    image_w=img.shape[1],
                    channels=channels,
                    kernel_type=kname,
                    kernel_size=int(kernel.shape[0]),
                    run_count=runs,
                    warmup_count=warmup,
                    mean_ms=mean_ms,
                    median_ms=float(median(timings)),
                    std_ms=float(np.std(np.array(timings, dtype=np.float32))),
                    min_ms=float(np.min(timings)),
                    max_ms=float(np.max(timings)),
                    throughput_mpix_s=float(mpix / (mean_ms / 1000.0)),
                    rank=0,
                    approximation_method="exact_2d_cuda",
                    reconstruction_error=0.0,
                )
                rows.append(row)

    return rows


def write_reports(rows: List[BenchmarkRow], output_reports_dir: Path) -> tuple[Path, Path]:
    output_reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_reports_dir / "benchmark_results.csv"
    json_path = output_reports_dir / "benchmark_results.json"

    import pandas as pd

    frame = pd.DataFrame([asdict(r) for r in rows])
    frame.to_csv(csv_path, index=False)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in rows], f, indent=2)

    return csv_path, json_path
