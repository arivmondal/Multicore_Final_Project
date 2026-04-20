from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_rgb_image(image: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(np.rint(np.asarray(image, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[2] == 3:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(output_path), array):
        raise RuntimeError(f"Failed to save image: {output_path}")


def save_difference_heatmap(reference: np.ndarray, test: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diff = np.abs(np.asarray(reference, dtype=np.float32) - np.asarray(test, dtype=np.float32))
    if diff.ndim == 3:
        diff = diff.mean(axis=2)
    vmax = max(float(np.max(diff)), 1e-6)
    plt.figure(figsize=(6, 5))
    plt.imshow(diff, cmap="inferno", vmin=0.0, vmax=vmax)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Absolute Difference Heatmap")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def save_singular_value_plot(singular_values: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(1, singular_values.size + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(x, singular_values, marker="o", linewidth=2)
    plt.title("Singular Values")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def save_cumulative_energy_plot(singular_values: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    energy = np.cumsum(np.square(singular_values))
    energy = energy / energy[-1]
    x = np.arange(1, singular_values.size + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(x, energy, marker="o", linewidth=2, color="#117733")
    plt.ylim(0.0, 1.02)
    plt.title("Cumulative Energy Retained")
    plt.xlabel("Rank")
    plt.ylabel("Energy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def save_runtime_plot(rows: list[dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranks = [row["rank"] for row in rows]
    runtimes = [row["low_rank_runtime_ms"] for row in rows]
    plt.figure(figsize=(6, 4))
    plt.plot(ranks, runtimes, marker="o", linewidth=2, color="#004488")
    plt.title("Low-Rank Runtime vs Rank")
    plt.xlabel("Rank")
    plt.ylabel("Runtime (ms)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def save_quality_plot(rows: list[dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranks = [row["rank"] for row in rows]
    psnr = [row["psnr"] for row in rows]
    ssim = [row["ssim"] for row in rows]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(ranks, psnr, marker="o", linewidth=2, color="#AA3377")
    ax1.set_xlabel("Rank")
    ax1.set_ylabel("PSNR (dB)", color="#AA3377")
    ax1.tick_params(axis="y", labelcolor="#AA3377")

    ax2 = ax1.twinx()
    ax2.plot(ranks, ssim, marker="s", linewidth=2, color="#228833")
    ax2.set_ylabel("SSIM", color="#228833")
    ax2.tick_params(axis="y", labelcolor="#228833")
    fig.suptitle("Approximation Quality vs Rank")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
