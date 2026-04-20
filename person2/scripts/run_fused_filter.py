import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from generate_kernels import disk_kernel, gaussian_kernel, motion_diag_kernel


SCRIPT_DIR = Path(__file__).resolve().parent
PERSON2_DIR = SCRIPT_DIR.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the fused person2/person3 CUDA pipeline on an image and display the result."
    )
    parser.add_argument("--input-image", required=True, help="Path to the source image file.")
    parser.add_argument(
        "--kernel-path",
        help="Optional path to a CSV kernel matrix. If omitted, --filter-type is used to generate one.",
    )
    parser.add_argument(
        "--filter-type",
        choices=["gaussian", "disk", "motion_diag"],
        help="Filter type to generate when --kernel-path is not provided.",
    )
    parser.add_argument("--kernel-size", type=int, required=True, help="Odd kernel size.")
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.0,
        help="Gaussian sigma. If omitted or <= 0, the generator default is used.",
    )
    parser.add_argument(
        "--disk-radius",
        type=float,
        default=0.0,
        help="Disk radius. If omitted or <= 0, a milder default of kernel_size / 3 is used.",
    )
    parser.add_argument(
        "--motion-thickness",
        type=int,
        default=1,
        help="Diagonal motion blur line thickness.",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        required=True,
        help="One or more ranks to evaluate, for example: --ranks 1 2 4 8",
    )
    parser.add_argument(
        "--kernel-name",
        help="Optional label for logs. Defaults to the filter type or kernel file stem.",
    )
    parser.add_argument(
        "--binary",
        help="Path to the fused CUDA binary. Defaults to person2/build/lowrank_svd(.exe).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PERSON2_DIR / "outputs" / "fused_preview"),
        help="Directory for temporary CSVs and saved filtered images.",
    )
    parser.add_argument(
        "--display-rank",
        type=int,
        help="Rank to display. Defaults to the largest requested rank.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip opening a matplotlib window and just save the images.",
    )
    return parser.parse_args()


def resolve_binary(binary_arg):
    if binary_arg:
        binary_path = Path(binary_arg).expanduser().resolve()
        if not binary_path.exists():
            raise FileNotFoundError(f"CUDA binary not found: {binary_path}")
        if sys.platform.startswith("win") and binary_path.suffix.lower() != ".exe":
            raise RuntimeError(
                f"On Windows, the CUDA binary must be a .exe file. "
                f"Got: {binary_path}. Rebuild person2 to produce lowrank_svd.exe."
            )
        return binary_path

    exe_candidate = PERSON2_DIR / "build" / "lowrank_svd.exe"
    non_windows_candidate = PERSON2_DIR / "build" / "lowrank_svd"

    if sys.platform.startswith("win"):
        if exe_candidate.exists():
            return exe_candidate.resolve()
        if non_windows_candidate.exists():
            raise RuntimeError(
                "Found person2/build/lowrank_svd, but not person2/build/lowrank_svd.exe. "
                "The existing file is likely a non-Windows build artifact. "
                "Build the CUDA binary on Windows so lowrank_svd.exe exists, or pass --binary "
                "to a valid .exe explicitly."
            )
        raise FileNotFoundError(
            "Could not find person2/build/lowrank_svd.exe. Build person2 first or pass --binary explicitly."
        )

    candidates = [exe_candidate, non_windows_candidate]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find the fused CUDA binary. Build person2 first or pass --binary explicitly."
    )


def load_input_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    if image.ndim == 2:
        return image.astype(np.float32) / 255.0
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if image.ndim == 3 and image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        return rgb.astype(np.float32) / 255.0
    raise ValueError(f"Unsupported image shape for {image_path}: {image.shape}")


def generate_kernel(filter_type, size, sigma, disk_radius, motion_thickness):
    if filter_type == "gaussian":
        actual_sigma = sigma if sigma > 0 else size / 6.0
        return gaussian_kernel(size, actual_sigma)
    if filter_type == "disk":
        actual_radius = disk_radius if disk_radius > 0 else None
        return disk_kernel(size, radius=actual_radius)
    return motion_diag_kernel(size, thickness=motion_thickness)


def resolve_kernel(args, work_dir):
    if args.kernel_path:
        kernel_path = Path(args.kernel_path).expanduser().resolve()
        if not kernel_path.exists():
            raise FileNotFoundError(f"Kernel CSV not found: {kernel_path}")
        kernel_name = args.kernel_name or kernel_path.stem
        return kernel_path, kernel_name

    if not args.filter_type:
        raise ValueError("Either --kernel-path or --filter-type is required.")

    kernel = generate_kernel(
        args.filter_type,
        args.kernel_size,
        args.sigma,
        args.disk_radius,
        args.motion_thickness,
    )
    kernel_path = work_dir / f"{args.filter_type}_{args.kernel_size}.csv"
    np.savetxt(kernel_path, kernel, delimiter=",", fmt="%.17g")
    kernel_name = args.kernel_name or args.filter_type
    return kernel_path, kernel_name


def save_filtered_preview(image_array, output_path):
    output_u8 = np.clip(np.rint(image_array * 255.0), 0, 255).astype(np.uint8)
    image_to_write = output_u8
    if output_u8.ndim == 3 and output_u8.shape[2] == 3:
        image_to_write = cv2.cvtColor(output_u8, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(output_path), image_to_write):
        raise RuntimeError(f"Failed to save preview image: {output_path}")


def get_image_channels(image):
    if image.ndim == 2:
        return [("gray", image)]
    return [("r", image[:, :, 0]), ("g", image[:, :, 1]), ("b", image[:, :, 2])]


def run_channel(binary_path, kernel_path, kernel_name, args, channel_name, channel_image, channel_output_dir, work_dir):
    input_csv_path = work_dir / f"{Path(args.input_image).stem}_{channel_name}.csv"
    np.savetxt(input_csv_path, channel_image, delimiter=",", fmt="%.9g")

    command = [
        str(binary_path),
        "--kernel_path",
        str(kernel_path),
        "--kernel_name",
        kernel_name,
        "--size",
        str(args.kernel_size),
        "--ranks",
    ]
    command.extend(str(rank) for rank in args.ranks)
    command.extend(
        [
            "--output_dir",
            str(channel_output_dir),
            "--image_path",
            str(input_csv_path),
            "--image_width",
            str(channel_image.shape[1]),
            "--image_height",
            str(channel_image.shape[0]),
        ]
    )

    print(f"Running fused CUDA pipeline for channel {channel_name}:")
    print(" ".join(command))
    subprocess.run(command, cwd=str(PERSON2_DIR), check=True)


def collect_rank_outputs(args, output_dir, channel_names):
    rank_outputs = {}
    for rank in args.ranks:
        if len(channel_names) == 1:
            filtered_csv = output_dir / f"rank_{rank}_filtered.csv"
            if not filtered_csv.exists():
                raise FileNotFoundError(f"Expected CUDA output was not produced: {filtered_csv}")
            rank_outputs[rank] = np.loadtxt(filtered_csv, delimiter=",")
            continue

        channels = []
        for channel_name in channel_names:
            filtered_csv = output_dir / f"channel_{channel_name}" / f"rank_{rank}_filtered.csv"
            if not filtered_csv.exists():
                raise FileNotFoundError(f"Expected CUDA output was not produced: {filtered_csv}")
            channels.append(np.loadtxt(filtered_csv, delimiter=","))
        rank_outputs[rank] = np.stack(channels, axis=-1)

    return rank_outputs


def main():
    args = parse_args()

    if args.kernel_size <= 0 or args.kernel_size % 2 == 0:
        raise ValueError("--kernel-size must be a positive odd integer.")
    if any(rank <= 0 for rank in args.ranks):
        raise ValueError("All ranks must be positive.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "pipeline_inputs"
    work_dir.mkdir(parents=True, exist_ok=True)

    binary_path = resolve_binary(args.binary)
    input_image_path = Path(args.input_image).expanduser().resolve()
    input_image = load_input_image(input_image_path)

    kernel_path, kernel_name = resolve_kernel(args, work_dir)
    channels = get_image_channels(input_image)
    channel_names = [channel_name for channel_name, _ in channels]

    if len(channels) == 1:
        run_channel(
            binary_path,
            kernel_path,
            kernel_name,
            args,
            channel_names[0],
            channels[0][1],
            output_dir,
            work_dir,
        )
    else:
        for channel_name, channel_image in channels:
            channel_output_dir = output_dir / f"channel_{channel_name}"
            channel_output_dir.mkdir(parents=True, exist_ok=True)
            run_channel(
                binary_path,
                kernel_path,
                kernel_name,
                args,
                channel_name,
                channel_image,
                channel_output_dir,
                work_dir,
            )

    display_rank = args.display_rank if args.display_rank is not None else max(args.ranks)
    if display_rank not in args.ranks:
        raise ValueError(f"--display-rank {display_rank} was not included in --ranks.")

    rank_outputs = collect_rank_outputs(args, output_dir, channel_names)
    for rank in args.ranks:
        filtered = rank_outputs[rank]
        preview_path = output_dir / f"rank_{rank}_filtered.png"
        save_filtered_preview(filtered, preview_path)
        print(f"saved rank {rank} preview to {preview_path}")

    chosen_filtered = rank_outputs[display_rank]
    if args.no_show:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    if input_image.ndim == 2:
        axes[0].imshow(input_image, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1].imshow(chosen_filtered, cmap="gray", vmin=0.0, vmax=1.0)
    else:
        axes[0].imshow(input_image, vmin=0.0, vmax=1.0)
        axes[1].imshow(chosen_filtered, vmin=0.0, vmax=1.0)
    axes[0].set_title("Input")
    axes[1].set_title(f"Filtered rank {display_rank}")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"run_fused_filter failed: {exc}", file=sys.stderr)
        sys.exit(1)
