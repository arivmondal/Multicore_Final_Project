from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REQUIRED_COLUMNS = {
    "kernel_type",
    "kernel_size",
    "image_h",
    "image_w",
    "mean_ms",
}


def _validate_columns(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"CSV is missing required columns: {missing_list}")


def _pixel_label(image_h: int, image_w: int) -> str:
    mpix = (image_h * image_w) / 1_000_000.0
    return f"{image_h}x{image_w} ({mpix:.2f} MP)"


def _plot_for_kernel_type(
    frame: pd.DataFrame,
    kernel_type: str,
    output_dir: Path,
    time_column: str,
    agg: str,
) -> Path:
    subset = frame[frame["kernel_type"] == kernel_type].copy()
    if subset.empty:
        raise ValueError(f"No rows found for kernel_type={kernel_type!r}")

    grouped = (
        subset.groupby(["kernel_size", "image_h", "image_w"], as_index=False)[time_column]
        .agg(agg)
        .sort_values(["image_h", "image_w", "kernel_size"])
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    for (image_h, image_w), chunk in grouped.groupby(["image_h", "image_w"], sort=True):
        label = _pixel_label(int(image_h), int(image_w))
        ax.plot(
            chunk["kernel_size"],
            chunk[time_column],
            marker="o",
            linewidth=2,
            label=label,
        )

    ax.set_title(f"Execution Time vs Kernel Size ({kernel_type})")
    ax.set_xlabel("Kernel Size")
    ax.set_ylabel("Execution Time (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Image Size")

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_kernel_type = kernel_type.replace(" ", "_")
    output_path = output_dir / f"exec_time_vs_kernel_{safe_kernel_type}.png"

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot execution time vs kernel size with overlays for image sizes, per filter type."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("output/reports/benchmark_results.csv"),
        help="Path to benchmark_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/reports/plots"),
        help="Directory where plots will be saved",
    )
    parser.add_argument(
        "--time-column",
        type=str,
        default="mean_ms",
        help="Time column to plot (e.g., mean_ms, median_ms, min_ms, max_ms)",
    )
    parser.add_argument(
        "--agg",
        choices=["mean", "median"],
        default="mean",
        help="Aggregation used if multiple rows share kernel_size and image size",
    )
    args = parser.parse_args()

    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")

    frame = pd.read_csv(args.csv_path)
    _validate_columns(frame)

    if args.time_column not in frame.columns:
        raise ValueError(f"Column {args.time_column!r} not found in CSV")

    outputs = []
    for kernel_type in sorted(frame["kernel_type"].dropna().unique()):
        output_path = _plot_for_kernel_type(
            frame=frame,
            kernel_type=str(kernel_type),
            output_dir=args.output_dir,
            time_column=args.time_column,
            agg=args.agg,
        )
        outputs.append(output_path)

    print("Saved plots:")
    for output in outputs:
        print(f"- {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
