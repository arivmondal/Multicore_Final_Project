import argparse
import os

import cv2
import numpy as np


def normalized(kernel):
    kernel = kernel.astype(np.float64)
    total = kernel.sum()
    if total <= 0:
        raise ValueError("Kernel sum must be positive.")
    return kernel / total


def gaussian_kernel(size, sigma):
    g = cv2.getGaussianKernel(size, sigma)
    return normalized(g @ g.T)


def disk_kernel(size, radius=None):
    kernel = np.zeros((size, size), dtype=np.float64)
    center = size // 2
    actual_radius = radius if radius is not None and radius > 0 else size / 3.0
    cv2.circle(kernel, (center, center), int(round(actual_radius)), 1.0, thickness=-1)
    return normalized(kernel)


def motion_diag_kernel(size, thickness=1):
    kernel = np.zeros((size, size), dtype=np.float64)
    cv2.line(kernel, (0, 0), (size - 1, size - 1), 1.0, thickness=max(1, int(thickness)))
    return normalized(kernel)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True, choices=["gaussian", "disk", "motion_diag"])
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument(
        "--radius",
        type=float,
        default=0.0,
        help="Disk radius. If omitted or <= 0, a milder default of size / 3 is used.",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=1,
        help="Diagonal motion blur line thickness.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.size <= 0 or args.size % 2 == 0:
        raise ValueError("--size must be a positive odd integer.")

    if args.kernel == "gaussian":
        sigma = args.sigma if args.sigma > 0 else args.size / 6.0
        kernel = gaussian_kernel(args.size, sigma)
    elif args.kernel == "disk":
        kernel = disk_kernel(args.size, radius=args.radius)
    else:
        kernel = motion_diag_kernel(args.size, thickness=args.thickness)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savetxt(args.output, kernel, delimiter=",", fmt="%.17g")
    print(f"saved {args.kernel} kernel to {args.output}")


if __name__ == "__main__":
    main()
