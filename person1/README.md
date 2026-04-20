# Task 1: Exact 2-D CUDA Baseline + Evaluation

This task implements Person 1 deliverables:
- Exact nonseparable 2-D convolution baseline in CUDA (GPU)
- Benchmarking across image/kernel sizes

## Structure

- `src/`: implementation code
- `input/images/`: source images for experiments
- `output/images/`: filtered output images
- `output/reports/`: CSV and JSON benchmark reports

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run

From `person1`:

```powershell
python -m src.cli
```

This command:
- Loads images from `input/images`
- Uses synthetic fallback images if input folder is empty
- Runs exact CUDA 2-D convolution with disk and diagonal-motion kernels
- Writes reports to `output/reports`
- Saves filtered images to `output/images`

## Test images

The repository includes three small sample inputs in `input/images`:
- `sample_gradient.png`
- `sample_checkerboard.png`
- `sample_motion.png`

These are enough to smoke-test the baseline without providing your own images.

## Example commands

From the project root (`Multicore_Final_Project`):

```powershell
# 1) Go to person1 folder
cd .\person1

# 2) (Optional) Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run default benchmark (512/1024/2048 and kernels 15/31/63)
python -m src.cli

# 5) Run a quicker benchmark
python -m src.cli --warmup 1 --runs 2

# 6) Run benchmark using explicit input/output folders
python -m src.cli --input-dir input/images --output-dir output --warmup 2 --runs 5
```

If you are already inside `person1`, run:

```powershell
pip install -r requirements.txt
python -m src.cli
```

Outputs are written to:
- `output/images/` for filtered images
- `output/reports/benchmark_results.csv` for tabular results
- `output/reports/benchmark_results.json` for JSON results

## Notes

- Border mode is fixed to replicate in the CUDA path.
- Internal numeric type is float32 in [0, 1].
- If CUDA is unavailable, the benchmark command will report the issue.
