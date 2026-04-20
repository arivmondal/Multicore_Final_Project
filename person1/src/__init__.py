"""Task 1 package for CUDA exact 2-D convolution baseline."""

from .integration_api import compute_metrics_from_paths, run_exact_cuda_from_paths

__all__ = [
	"run_exact_cuda_from_paths",
	"compute_metrics_from_paths",
]
