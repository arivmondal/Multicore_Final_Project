import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["rank"] = int(row["rank"])
        row["frobenius_error"] = float(row["frobenius_error"])
        row["relative_frobenius_error"] = float(row["relative_frobenius_error"])
        row["cumulative_energy_retained"] = float(row["cumulative_energy_retained"])
    return rows


def maybe_2d(array):
    array = np.asarray(array)
    return array if array.ndim > 1 else array.reshape(1, -1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--max_reconstructions", type=int, default=4)
    args = parser.parse_args()

    metrics = load_metrics(os.path.join(args.run_dir, "metrics.csv"))
    singular_values = np.loadtxt(os.path.join(args.run_dir, "singular_values.csv"), delimiter=",")
    singular_values = np.ravel(singular_values)

    plt.figure()
    plt.plot(np.arange(1, singular_values.size + 1), singular_values, marker="o")
    plt.xlabel("Index")
    plt.ylabel("Singular value")
    plt.title("Singular values")
    plt.tight_layout()
    plt.savefig(os.path.join(args.run_dir, "singular_values.png"), dpi=150)
    plt.close()

    ranks = [row["rank"] for row in metrics]
    relative_errors = [row["relative_frobenius_error"] for row in metrics]

    plt.figure()
    plt.plot(ranks, relative_errors, marker="o")
    plt.xlabel("Rank")
    plt.ylabel("Relative Frobenius error")
    plt.title("Reconstruction error vs rank")
    plt.tight_layout()
    plt.savefig(os.path.join(args.run_dir, "reconstruction_error.png"), dpi=150)
    plt.close()

    original = np.loadtxt(os.path.join(args.run_dir, "input_kernel.csv"), delimiter=",")
    original = maybe_2d(original)
    shown = metrics[: max(1, min(args.max_reconstructions, len(metrics)))]

    fig, axes = plt.subplots(1 + len(shown), 2, figsize=(8, 3 * (1 + len(shown))))
    axes = np.atleast_2d(axes)

    axes[0, 0].imshow(original, cmap="viridis")
    axes[0, 0].set_title("Original kernel")
    axes[0, 1].axis("off")

    for i, row in enumerate(shown, start=1):
        rank = row["rank"]
        reconstruction = np.loadtxt(
            os.path.join(args.run_dir, f"rank_{rank}_reconstruction.csv"),
            delimiter=",",
        )
        reconstruction = maybe_2d(reconstruction)
        diff = original - reconstruction
        axes[i, 0].imshow(reconstruction, cmap="viridis")
        axes[i, 0].set_title(f"Rank {rank} reconstruction")
        axes[i, 1].imshow(diff, cmap="coolwarm")
        axes[i, 1].set_title(f"Rank {rank} error")

    for row_axes in axes:
        for ax in row_axes:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(args.run_dir, "reconstructions.png"), dpi=150)
    plt.close()

    print(f"saved plots in {args.run_dir}")


if __name__ == "__main__":
    main()
