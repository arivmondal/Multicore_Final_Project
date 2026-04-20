from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from django.conf import settings

from .kernel_factory import FILTER_LABELS, build_kernel, kernel_parameter_summary
from .metrics import compute_image_metrics, compute_kernel_rank_metrics, to_float01
from .plotting import (
    save_cumulative_energy_plot,
    save_difference_heatmap,
    save_rgb_image,
    save_singular_value_plot,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = settings.MEDIA_ROOT / "runs"
PERSON1_CONVOLUTION_PATH = REPO_ROOT / "person1" / "src" / "convolution_runner.py"
LOW_RANK_SCRIPT = REPO_ROOT / "person2" / "scripts" / "run_fused_filter.py"
LOW_RANK_BINARY = REPO_ROOT / "person2" / "build" / ("lowrank_svd.exe" if os.name == "nt" else "lowrank_svd")
CACHE_SCHEMA_VERSION = "v2"

DEMO_IMAGES = [
    {
        "slug": "galaxy",
        "label": "Milky Way Galaxy",
        "path": REPO_ROOT / "person2" / "inputs" / "galaxy.png",
    },
    {
        "slug": "lights",
        "label": "City Lights",
        "path": REPO_ROOT / "person2" / "inputs" / "lights.png",
    },
    {
        "slug": "dots",
        "label": "Dots Pattern",
        "path": REPO_ROOT / "person2" / "inputs" / "dots.png",
    },
    {
        "slug": "eye",
        "label": "Eye",
        "path": REPO_ROOT / "person2" / "inputs" / "eye.png",
    },
    {
        "slug": "sample-gradient",
        "label": "Gradient",
        "path": REPO_ROOT / "person1" / "input" / "images" / "sample_gradient.png",
    },
    {
        "slug": "sample-motion",
        "label": "Motion Sample",
        "path": REPO_ROOT / "person1" / "input" / "images" / "sample_motion.png",
    },
    {
        "slug": "sample-checkerboard",
        "label": "Checkerboard",
        "path": REPO_ROOT / "person1" / "input" / "images" / "sample_checkerboard.png",
    },
]


def get_demo_image_choices() -> list[tuple[str, str]]:
    return [(item["slug"], item["label"]) for item in DEMO_IMAGES if item["path"].exists()]


def get_demo_images() -> list[dict[str, str]]:
    return [
        {
            "slug": item["slug"],
            "label": item["label"],
            "exists": str(item["path"].exists()).lower(),
        }
        for item in DEMO_IMAGES
        if item["path"].exists()
    ]


def get_sample_path(sample_slug: str) -> Path:
    for item in DEMO_IMAGES:
        if item["slug"] == sample_slug:
            return item["path"]
    raise FileNotFoundError(f"Unknown demo image slug: {sample_slug}")


def media_url_for(path: Path) -> str:
    relative = path.resolve().relative_to(settings.MEDIA_ROOT.resolve())
    return f"{settings.MEDIA_URL}{relative.as_posix()}"


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _person1_convolution_module():
    return _load_module("person1_convolution_runner", PERSON1_CONVOLUTION_PATH)


def configure_runtime_environment() -> None:
    cache_dir = settings.MEDIA_ROOT / "_runtime" / "cupy_cache"
    tmp_dir = settings.MEDIA_ROOT / "_runtime" / "tmp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUPY_CACHE_DIR", str(cache_dir))
    os.environ.setdefault("TEMP", str(tmp_dir))
    os.environ.setdefault("TMP", str(tmp_dir))


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Failed to decode the selected image.")
    if image.ndim == 2:
        return (image.astype(np.float32) / 255.0).astype(np.float32)
    if image.ndim == 3 and image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return (rgb.astype(np.float32) / 255.0).astype(np.float32)
    if image.ndim == 3 and image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        return (rgb.astype(np.float32) / 255.0).astype(np.float32)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _save_input_bytes(input_bytes: bytes, file_suffix: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(input_bytes)
    if file_suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
        converted = decode_image_bytes(input_bytes)
        save_rgb_image(converted, output_path.with_suffix(".png"))


def _stable_request_hash(input_bytes: bytes, params: dict[str, Any], prefix: str) -> str:
    payload = json.dumps(
        {"prefix": prefix, "params": params, "cache_schema_version": CACHE_SCHEMA_VERSION},
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    digest = hashlib.sha256(input_bytes + payload).hexdigest()
    return digest[:16]


def resolve_input_source(uploaded_file, sample_slug: str | None) -> dict[str, Any]:
    if uploaded_file:
        file_bytes = uploaded_file.read()
        suffix = Path(uploaded_file.name).suffix or ".png"
        label = uploaded_file.name
        source_kind = "upload"
    else:
        sample_path = get_sample_path(sample_slug or "")
        file_bytes = sample_path.read_bytes()
        suffix = sample_path.suffix or ".png"
        label = sample_path.name
        source_kind = "sample"
    return {
        "bytes": file_bytes,
        "suffix": suffix,
        "label": label,
        "kind": source_kind,
    }


def _parse_lowrank_metrics(metrics_path: Path) -> dict[int, dict[str, float]]:
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed: dict[int, dict[str, float]] = {}
    for row in rows:
        rank = int(row["rank"])
        parsed[rank] = {
            "rank": rank,
            "frobenius_error": float(row["frobenius_error"]),
            "relative_frobenius_error": float(row["relative_frobenius_error"]),
            "cumulative_energy_retained": float(row["cumulative_energy_retained"]),
            "filter_time_ms": float(row["filter_time_ms"]),
        }
    return parsed


def _load_lowrank_metrics(output_dir: Path) -> dict[int, dict[str, float]]:
    top_level_metrics = output_dir / "metrics.csv"
    if top_level_metrics.exists():
        return _parse_lowrank_metrics(top_level_metrics)

    channel_dirs = sorted(path for path in output_dir.glob("channel_*") if path.is_dir())
    if not channel_dirs:
        raise FileNotFoundError(f"No low-rank metrics were produced under {output_dir}")

    aggregated: dict[int, dict[str, float]] = {}
    for channel_dir in channel_dirs:
        channel_metrics_path = channel_dir / "metrics.csv"
        if not channel_metrics_path.exists():
            raise FileNotFoundError(f"Missing per-channel metrics file: {channel_metrics_path}")
        channel_metrics = _parse_lowrank_metrics(channel_metrics_path)
        for rank, values in channel_metrics.items():
            entry = aggregated.setdefault(
                rank,
                {
                    "rank": rank,
                    "frobenius_error": values["frobenius_error"],
                    "relative_frobenius_error": values["relative_frobenius_error"],
                    "cumulative_energy_retained": values["cumulative_energy_retained"],
                    "filter_time_ms": 0.0,
                },
            )
            entry["filter_time_ms"] += values["filter_time_ms"]
    return aggregated


def _load_result_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read generated image: {image_path}")
    if image.ndim == 2:
        return (image.astype(np.float32) / 255.0).astype(np.float32)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 255.0).astype(np.float32)


def _convolve_cpu_exact(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv2.filter2D(np.asarray(image, dtype=np.float32), -1, kernel, borderType=cv2.BORDER_REPLICATE)


def _convolve_cpu_low_rank(image: np.ndarray, kernel: np.ndarray, rank: int) -> np.ndarray:
    kernel64 = np.asarray(kernel, dtype=np.float64)
    u, singular_values, vt = np.linalg.svd(kernel64, full_matrices=False)
    result = np.zeros_like(np.asarray(image, dtype=np.float32))
    for component in range(rank):
        row_weights = vt[component, :].astype(np.float32)
        col_weights = (singular_values[component] * u[:, component]).astype(np.float32)
        filtered = cv2.sepFilter2D(
            np.asarray(image, dtype=np.float32),
            -1,
            row_weights,
            col_weights,
            borderType=cv2.BORDER_REPLICATE,
        )
        result += filtered
    return result


def run_exact_baseline(image: np.ndarray, kernel: np.ndarray, output_path: Path) -> tuple[dict[str, Any], list[str]]:
    configure_runtime_environment()
    warnings: list[str] = []
    start = time.perf_counter()
    kernel_time_ms: float | None = None
    device_label = "CPU / OpenCV"

    try:
        module = _person1_convolution_module()
        convolver = module.CudaExactConvolver(border_mode="replicate")
        kernel_time_ms = float(convolver.benchmark_once_ms(image, kernel))
        filtered = to_float01(convolver.convolve(image, kernel))
        device_label = "GPU / CuPy Raw CUDA"
    except Exception as exc:
        warnings.append(f"Exact CUDA baseline unavailable, using OpenCV CPU fallback: {exc}")
        filtered = to_float01(_convolve_cpu_exact(image, kernel))

    end_to_end_ms = (time.perf_counter() - start) * 1000.0
    save_rgb_image(filtered, output_path)
    return {
        "output_image": filtered,
        "output_path": str(output_path),
        "output_url": media_url_for(output_path),
        "end_to_end_runtime_ms": end_to_end_ms,
        "kernel_runtime_ms": kernel_time_ms,
        "device": device_label,
    }, warnings


def run_low_rank_filter(
    input_image_path: Path,
    image: np.ndarray,
    filter_type: str,
    kernel_size: int,
    rank_list: list[int],
    sigma: float | None,
    disk_radius: float | None,
    motion_thickness: int,
    output_dir: Path,
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    configure_runtime_environment()
    warnings: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(LOW_RANK_SCRIPT),
        "--input-image",
        str(input_image_path),
        "--filter-type",
        filter_type,
        "--kernel-size",
        str(kernel_size),
        "--ranks",
    ]
    command.extend(str(rank) for rank in rank_list)
    command.extend(["--output-dir", str(output_dir), "--no-show"])

    if filter_type == "gaussian" and sigma and sigma > 0:
        command.extend(["--sigma", str(sigma)])
    if filter_type == "disk" and disk_radius and disk_radius > 0:
        command.extend(["--disk-radius", str(disk_radius)])
    if filter_type == "motion_diag" and motion_thickness > 0:
        command.extend(["--motion-thickness", str(motion_thickness)])

    device_label = "GPU / lowrank_svd"
    start = time.perf_counter()

    try:
        if not LOW_RANK_BINARY.exists():
            raise FileNotFoundError(f"Low-rank CUDA binary not found: {LOW_RANK_BINARY}")
        subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        run_time_ms = (time.perf_counter() - start) * 1000.0
        metrics_by_rank = _load_lowrank_metrics(output_dir)
        results: dict[int, dict[str, Any]] = {}
        for rank in rank_list:
            image_path = output_dir / f"rank_{rank}_filtered.png"
            result_image = _load_result_image(image_path)
            rank_metrics = metrics_by_rank.get(rank, {}).copy()
            rank_metrics.update(
                {
                    "output_image": result_image,
                    "output_path": str(image_path),
                    "output_url": media_url_for(image_path),
                    "end_to_end_runtime_ms": run_time_ms,
                    "device": device_label,
                }
            )
            results[rank] = rank_metrics
        return results, warnings
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout or str(exc)
        warnings.append(f"Low-rank CUDA path unavailable, using CPU separable fallback: {detail}")
        device_label = "CPU / OpenCV separable fallback"
        kernel = build_kernel(
            filter_type=filter_type,
            kernel_size=kernel_size,
            sigma=sigma,
            disk_radius=disk_radius,
            motion_thickness=motion_thickness,
        )
        results = {}
        for rank in rank_list:
            start_rank = time.perf_counter()
            output = to_float01(_convolve_cpu_low_rank(image, kernel, rank))
            runtime_ms = (time.perf_counter() - start_rank) * 1000.0
            image_path = output_dir / f"rank_{rank}_filtered.png"
            save_rgb_image(output, image_path)
            kernel_metrics = compute_kernel_rank_metrics(kernel, rank)
            results[rank] = {
                "rank": rank,
                "frobenius_error": kernel_metrics["frobenius_error"],
                "relative_frobenius_error": kernel_metrics["relative_frobenius_error"],
                "cumulative_energy_retained": kernel_metrics["cumulative_energy_retained"],
                "filter_time_ms": runtime_ms,
                "end_to_end_runtime_ms": runtime_ms,
                "output_image": output,
                "output_path": str(image_path),
                "output_url": media_url_for(image_path),
                "device": device_label,
            }
        return results, warnings
    except Exception as exc:
        warnings.append(f"Low-rank CUDA path unavailable, using CPU separable fallback: {exc}")
        device_label = "CPU / OpenCV separable fallback"
        kernel = build_kernel(
            filter_type=filter_type,
            kernel_size=kernel_size,
            sigma=sigma,
            disk_radius=disk_radius,
            motion_thickness=motion_thickness,
        )
        results = {}
        for rank in rank_list:
            start_rank = time.perf_counter()
            output = to_float01(_convolve_cpu_low_rank(image, kernel, rank))
            runtime_ms = (time.perf_counter() - start_rank) * 1000.0
            image_path = output_dir / f"rank_{rank}_filtered.png"
            save_rgb_image(output, image_path)
            kernel_metrics = compute_kernel_rank_metrics(kernel, rank)
            results[rank] = {
                "rank": rank,
                "frobenius_error": kernel_metrics["frobenius_error"],
                "relative_frobenius_error": kernel_metrics["relative_frobenius_error"],
                "cumulative_energy_retained": kernel_metrics["cumulative_energy_retained"],
                "filter_time_ms": runtime_ms,
                "end_to_end_runtime_ms": runtime_ms,
                "output_image": output,
                "output_path": str(image_path),
                "output_url": media_url_for(image_path),
                "device": device_label,
            }
        return results, warnings


def _serialize_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    serializable = json.loads(json.dumps(payload, default=str))
    return serializable


def _load_cached_metadata(metadata_path: Path) -> dict[str, Any]:
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def get_system_info() -> dict[str, Any]:
    configure_runtime_environment()
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "low_rank_binary": str(LOW_RANK_BINARY.exists()).lower(),
        "low_rank_binary_path": str(LOW_RANK_BINARY),
        "cupy_cuda_available": "false",
        "cuda_device_name": "Unavailable",
    }
    try:
        import cupy as cp

        count = cp.cuda.runtime.getDeviceCount()
        if count > 0:
            props = cp.cuda.runtime.getDeviceProperties(0)
            device_name = props["name"].decode("utf-8") if isinstance(props["name"], bytes) else str(props["name"])
            info["cupy_cuda_available"] = "true"
            info["cuda_device_name"] = device_name
    except Exception:
        pass
    return info


def process_demo(form_data: dict[str, Any]) -> dict[str, Any]:
    source = resolve_input_source(form_data.get("input_image"), form_data.get("sample_image"))
    sigma = form_data.get("sigma")
    disk_radius = form_data.get("disk_radius")
    motion_thickness = form_data.get("motion_thickness") or 1
    params = {
        "filter_type": form_data["filter_type"],
        "kernel_size": int(form_data["kernel_size"]),
        "rank": int(form_data["rank"]),
        "run_mode": form_data["run_mode"],
        "sigma": float(sigma) if sigma is not None else None,
        "disk_radius": float(disk_radius) if disk_radius is not None else None,
        "motion_thickness": int(motion_thickness),
    }

    run_key = _stable_request_hash(source["bytes"], params, "demo")
    run_dir = RUNS_ROOT / run_key
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        return _load_cached_metadata(metadata_path)

    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = run_dir / f"input{source['suffix']}"
    input_path.write_bytes(source["bytes"])

    image = decode_image_bytes(source["bytes"])
    kernel = build_kernel(
        filter_type=params["filter_type"],
        kernel_size=params["kernel_size"],
        sigma=params["sigma"],
        disk_radius=params["disk_radius"],
        motion_thickness=params["motion_thickness"],
    )
    kernel_metrics = compute_kernel_rank_metrics(kernel, params["rank"])

    singular_plot_path = run_dir / "singular_values.png"
    energy_plot_path = run_dir / "cumulative_energy.png"
    save_singular_value_plot(np.asarray(kernel_metrics["singular_values"]), singular_plot_path)
    save_cumulative_energy_plot(np.asarray(kernel_metrics["singular_values"]), energy_plot_path)

    original_path = run_dir / "original.png"
    save_rgb_image(image, original_path)

    warnings: list[str] = []
    baseline_result = None
    if params["run_mode"] in {"baseline", "both"}:
        baseline_result, baseline_warnings = run_exact_baseline(image, kernel, run_dir / "baseline.png")
        warnings.extend(baseline_warnings)

    lowrank_result = None
    if params["run_mode"] in {"low_rank", "both"}:
        lowrank_results, lowrank_warnings = run_low_rank_filter(
            input_image_path=input_path,
            image=image,
            filter_type=params["filter_type"],
            kernel_size=params["kernel_size"],
            rank_list=[params["rank"]],
            sigma=params["sigma"],
            disk_radius=params["disk_radius"],
            motion_thickness=params["motion_thickness"],
            output_dir=run_dir / "lowrank",
        )
        warnings.extend(lowrank_warnings)
        lowrank_result = lowrank_results[params["rank"]]

    quality_metrics = None
    diff_heatmap_url = None
    speedup = None
    if baseline_result and lowrank_result:
        quality_metrics = compute_image_metrics(baseline_result["output_image"], lowrank_result["output_image"])
        diff_heatmap_path = run_dir / "difference_heatmap.png"
        save_difference_heatmap(baseline_result["output_image"], lowrank_result["output_image"], diff_heatmap_path)
        diff_heatmap_url = media_url_for(diff_heatmap_path)
        if lowrank_result["filter_time_ms"] > 0:
            speedup = baseline_result["end_to_end_runtime_ms"] / lowrank_result["filter_time_ms"]

    payload = {
        "run_key": run_key,
        "source_label": source["label"],
        "warnings": warnings,
        "original_url": media_url_for(original_path),
        "baseline": {
            "available": baseline_result is not None,
            "output_url": baseline_result["output_url"] if baseline_result else None,
            "end_to_end_runtime_ms": baseline_result["end_to_end_runtime_ms"] if baseline_result else None,
            "kernel_runtime_ms": baseline_result["kernel_runtime_ms"] if baseline_result else None,
            "device": baseline_result["device"] if baseline_result else None,
        },
        "low_rank": {
            "available": lowrank_result is not None,
            "output_url": lowrank_result["output_url"] if lowrank_result else None,
            "end_to_end_runtime_ms": lowrank_result["end_to_end_runtime_ms"] if lowrank_result else None,
            "filter_time_ms": lowrank_result["filter_time_ms"] if lowrank_result else None,
            "device": lowrank_result["device"] if lowrank_result else None,
        },
        "quality_metrics": quality_metrics,
        "kernel_metrics": {
            "frobenius_error": kernel_metrics["frobenius_error"],
            "relative_frobenius_error": kernel_metrics["relative_frobenius_error"],
            "cumulative_energy_retained": kernel_metrics["cumulative_energy_retained"],
        },
        "kernel_plots": {
            "singular_values_url": media_url_for(singular_plot_path),
            "cumulative_energy_url": media_url_for(energy_plot_path),
        },
        "difference_heatmap_url": diff_heatmap_url,
        "speedup": speedup,
        "image_dimensions": {
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
            "channels": 1 if image.ndim == 2 else int(image.shape[2]),
        },
        "parameters": kernel_parameter_summary(
            filter_type=params["filter_type"],
            kernel_size=params["kernel_size"],
            sigma=params["sigma"],
            disk_radius=params["disk_radius"],
            motion_thickness=params["motion_thickness"],
        )
        | {
            "rank": params["rank"],
            "run_mode": params["run_mode"],
        },
    }
    metadata_path.write_text(json.dumps(_serialize_result_payload(payload), indent=2), encoding="utf-8")
    return payload
