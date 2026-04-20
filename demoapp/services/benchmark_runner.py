from __future__ import annotations

import csv
import json
from typing import Any

from .filter_runner import (
    RUNS_ROOT,
    decode_image_bytes,
    media_url_for,
    resolve_input_source,
    run_exact_baseline,
    run_low_rank_filter,
    _stable_request_hash,
)
from .kernel_factory import build_kernel, kernel_parameter_summary
from .metrics import compute_image_metrics, compute_kernel_rank_metrics
from .plotting import (
    save_cumulative_energy_plot,
    save_quality_plot,
    save_rgb_image,
    save_runtime_plot,
    save_singular_value_plot,
)


def parse_rank_sweep(rank_text: str, fallback_rank: int) -> list[int]:
    ranks = []
    for item in rank_text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        ranks.append(int(stripped))
    if not ranks:
        ranks = [fallback_rank]
    return sorted(set(ranks))


def _best_tradeoff(rows: list[dict[str, float]]) -> dict[str, float] | None:
    if not rows:
        return None
    high_quality = [row for row in rows if row["ssim"] >= 0.95]
    if high_quality:
        return max(high_quality, key=lambda row: row["speedup"])
    return max(rows, key=lambda row: row["speedup"] * max(row["ssim"], 1e-6))


def run_benchmark(form_data: dict[str, Any]) -> dict[str, Any]:
    source = resolve_input_source(form_data.get("input_image"), form_data.get("sample_image"))
    sigma = form_data.get("sigma")
    disk_radius = form_data.get("disk_radius")
    motion_thickness = form_data.get("motion_thickness") or 1
    ranks = parse_rank_sweep(form_data.get("benchmark_ranks", ""), int(form_data["rank"]))
    params = {
        "filter_type": form_data["filter_type"],
        "kernel_size": int(form_data["kernel_size"]),
        "ranks": ranks,
        "sigma": float(sigma) if sigma is not None else None,
        "disk_radius": float(disk_radius) if disk_radius is not None else None,
        "motion_thickness": int(motion_thickness),
    }

    run_key = _stable_request_hash(source["bytes"], params, "benchmark")
    run_dir = RUNS_ROOT / run_key
    metadata_path = run_dir / "benchmark_metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = run_dir / f"input{source['suffix']}"
    input_path.write_bytes(source["bytes"])
    image = decode_image_bytes(source["bytes"])
    save_rgb_image(image, run_dir / "original.png")

    kernel = build_kernel(
        filter_type=params["filter_type"],
        kernel_size=params["kernel_size"],
        sigma=params["sigma"],
        disk_radius=params["disk_radius"],
        motion_thickness=params["motion_thickness"],
    )
    kernel_metrics_by_rank = {rank: compute_kernel_rank_metrics(kernel, rank) for rank in ranks}
    save_singular_value_plot(kernel_metrics_by_rank[ranks[0]]["singular_values"], run_dir / "singular_values.png")
    save_cumulative_energy_plot(kernel_metrics_by_rank[ranks[0]]["singular_values"], run_dir / "cumulative_energy.png")

    baseline_result, warnings = run_exact_baseline(image, kernel, run_dir / "baseline.png")
    lowrank_results, lowrank_warnings = run_low_rank_filter(
        input_image_path=input_path,
        image=image,
        filter_type=params["filter_type"],
        kernel_size=params["kernel_size"],
        rank_list=ranks,
        sigma=params["sigma"],
        disk_radius=params["disk_radius"],
        motion_thickness=params["motion_thickness"],
        output_dir=run_dir / "lowrank",
    )
    warnings.extend(lowrank_warnings)

    table_rows: list[dict[str, float]] = []
    csv_path = run_dir / "benchmark_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "low_rank_runtime_ms",
                "baseline_runtime_ms",
                "speedup",
                "mse",
                "psnr",
                "ssim",
                "max_abs_difference",
                "frobenius_error",
                "relative_frobenius_error",
                "cumulative_energy_retained",
            ],
        )
        writer.writeheader()
        for rank in ranks:
            lowrank = lowrank_results[rank]
            quality = compute_image_metrics(baseline_result["output_image"], lowrank["output_image"])
            speedup = 0.0
            if lowrank["filter_time_ms"] > 0:
                speedup = baseline_result["end_to_end_runtime_ms"] / lowrank["filter_time_ms"]
            row = {
                "rank": rank,
                "low_rank_runtime_ms": lowrank["filter_time_ms"],
                "baseline_runtime_ms": baseline_result["end_to_end_runtime_ms"],
                "speedup": speedup,
                "mse": quality["mse"],
                "psnr": quality["psnr"],
                "ssim": quality["ssim"],
                "max_abs_difference": quality["max_abs_difference"],
                "frobenius_error": kernel_metrics_by_rank[rank]["frobenius_error"],
                "relative_frobenius_error": kernel_metrics_by_rank[rank]["relative_frobenius_error"],
                "cumulative_energy_retained": kernel_metrics_by_rank[rank]["cumulative_energy_retained"],
            }
            writer.writerow(row)
            table_rows.append(row)

    save_runtime_plot(table_rows, run_dir / "runtime_vs_rank.png")
    save_quality_plot(table_rows, run_dir / "quality_vs_rank.png")
    best_tradeoff = _best_tradeoff(table_rows)

    payload = {
        "run_key": run_key,
        "warnings": warnings,
        "source_label": source["label"],
        "baseline": {
            "output_url": baseline_result["output_url"],
            "device": baseline_result["device"],
            "end_to_end_runtime_ms": baseline_result["end_to_end_runtime_ms"],
            "kernel_runtime_ms": baseline_result["kernel_runtime_ms"],
        },
        "rows": table_rows,
        "row_count": len(table_rows),
        "csv_url": media_url_for(csv_path),
        "plots": {
            "singular_values_url": media_url_for(run_dir / "singular_values.png"),
            "cumulative_energy_url": media_url_for(run_dir / "cumulative_energy.png"),
            "runtime_vs_rank_url": media_url_for(run_dir / "runtime_vs_rank.png"),
            "quality_vs_rank_url": media_url_for(run_dir / "quality_vs_rank.png"),
        },
        "parameters": kernel_parameter_summary(
            filter_type=params["filter_type"],
            kernel_size=params["kernel_size"],
            sigma=params["sigma"],
            disk_radius=params["disk_radius"],
            motion_thickness=params["motion_thickness"],
        )
        | {"ranks": ranks},
        "best_tradeoff": best_tradeoff,
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
