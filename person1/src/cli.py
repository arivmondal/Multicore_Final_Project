from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import run_benchmark, write_reports
from .convolution_runner import CudaExactConvolver, CudaUnavailableError
from .io_utils import ensure_output_dirs, load_images_or_synthetic


def run_benchmark_cmd(input_dir: Path, output_dir: Path, warmup: int, runs: int) -> int:
    out_images, out_reports = ensure_output_dirs(output_dir)
    images = load_images_or_synthetic(input_dir)

    try:
        convolver = CudaExactConvolver(border_mode="replicate")
    except CudaUnavailableError as exc:
        print(f"CUDA setup error: {exc}")
        return 2

    rows = run_benchmark(
        convolver=convolver,
        images=images,
        image_sizes=(512, 1024, 2048),
        kernel_sizes=(15, 31, 63),
        warmup=warmup,
        runs=runs,
        output_images_dir=out_images,
    )
    csv_path, json_path = write_reports(rows, out_reports)
    print(f"Benchmark completed. CSV: {csv_path}")
    print(f"Benchmark completed. JSON: {json_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Task1 CUDA exact 2-D convolution baseline")
    parser.add_argument("command", nargs="?", default="benchmark", choices=["benchmark"], help="Pipeline command")
    parser.add_argument("--input-dir", default="input/images", help="Input images folder")
    parser.add_argument("--output-dir", default="output", help="Output base folder")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs for benchmark")
    parser.add_argument("--runs", type=int, default=10, help="Measured runs for benchmark")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    return run_benchmark_cmd(
        input_dir=input_dir,
        output_dir=output_dir,
        warmup=args.warmup,
        runs=args.runs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
