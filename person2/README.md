# Person 2: Kernel SVD Export

Takes a 2-D kernel, run GPU SVD, and export separable row/column terms for Person 3.

For rank `r`, the truncated approximation is

`K_r(x,y) = sum_{i=0}^{r-1} sigma_i * u_i[x] * v_i[y]`

The exported terms are:
- `rank_r_rows.csv`: each row is `v_i`
- `rank_r_cols.csv`: each row is `sigma_i * u_i`

## Build

Requires CUDA 10.1, `nvcc`, `g++`, and cuSOLVER.

```bash
cd person2
make
```

If CUDA is not on the default path:

```bash
make CUDA_HOME=/usr/local/cuda-10.1
```

## Generate kernels

Requires Python 3, NumPy, and OpenCV.

```bash
python scripts/generate_kernels.py --kernel gaussian --size 31 --sigma 5.0 --output outputs/gaussian_31.csv
python scripts/generate_kernels.py --kernel disk --size 31 --output outputs/disk_31.csv
python scripts/generate_kernels.py --kernel motion_diag --size 31 --output outputs/motion_diag_31.csv
```

## Run low-rank approximation

```bash
./build/lowrank_svd --kernel_path outputs/gaussian_31.csv --kernel_name gaussian --size 31 --ranks 1 2 4 8 --output_dir outputs/gaussian_31
./build/lowrank_svd --kernel_path outputs/disk_31.csv --kernel_name disk --size 31 --ranks 1 2 4 8 --output_dir outputs/disk_31
./build/lowrank_svd --kernel_path outputs/motion_diag_31.csv --kernel_name motion_diag --size 31 --ranks 1 2 4 8 --output_dir outputs/motion_diag_31
```

Example sanity checks:
- Gaussian rank-1 should be exact or nearly exact.
- Disk rank-1 should not be exact.
- Diagonal motion rank-1 should not be exact.
- Error should drop as rank increases.

## Outputs

For each run directory:
- `input_kernel.csv`
- `singular_values.csv`
- `metrics.csv`
- `run_log.txt`
- `rank_r_reconstruction.csv`
- `rank_r_rows.csv`
- `rank_r_cols.csv`
- `rank_r_log.txt`

## Person 3 usage

For each component `i`:
- apply horizontal convolution with `v_i`
- apply vertical convolution with `sigma_i * u_i`
- sum the results over all exported components

The files are stored as CSV for easy loading. Each row in `rank_r_rows.csv` and `rank_r_cols.csv` is one component.

## Analysis

```bash
python scripts/analyze_results.py --run_dir outputs/gaussian_31
python scripts/analyze_results.py --run_dir outputs/disk_31
python scripts/analyze_results.py --run_dir outputs/motion_diag_31
```

This writes:
- `singular_values.png`
- `reconstruction_error.png`
- `reconstructions.png`
