from __future__ import annotations

from pathlib import Path

import numpy as np


class CudaUnavailableError(RuntimeError):
    pass


class CudaExactConvolver:
    def __init__(self, border_mode: str = "replicate") -> None:
        if border_mode != "replicate":
            raise ValueError("Current CUDA kernel supports only replicate border mode.")

        self.border_mode = border_mode
        try:
            import cupy as cp  # pylint: disable=import-outside-toplevel
        except Exception as exc:  # pragma: no cover - environment dependent
            raise CudaUnavailableError("CuPy is not available. Install a CUDA-enabled CuPy build.") from exc

        self.cp = cp
        self._kernel = self._load_kernel()

    def _load_kernel(self):
        cu_path = Path(__file__).with_name("cuda_convolution.cu")
        src = cu_path.read_text(encoding="utf-8")
        module = self.cp.RawModule(code=src, options=("--std=c++11",), name_expressions=("conv2d_exact",))
        return module.get_function("conv2d_exact")

    @staticmethod
    def _prepare_image(image: np.ndarray) -> np.ndarray:
        arr = image.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.ndim != 3:
            raise ValueError("Image must be HxW or HxWxC.")
        return np.ascontiguousarray(arr)

    @staticmethod
    def _prepare_kernel(kernel: np.ndarray) -> np.ndarray:
        if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
            raise ValueError("Kernel must be square 2-D.")
        if kernel.shape[0] % 2 == 0:
            raise ValueError("Kernel size must be odd.")
        ker = kernel.astype(np.float32)
        return np.ascontiguousarray(ker)

    def convolve(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        cp = self.cp
        img = self._prepare_image(image)
        ker = self._prepare_kernel(kernel)

        h, w, c = img.shape
        ksize = int(ker.shape[0])
        pad = ksize // 2

        d_input = cp.asarray(img)
        d_kernel = cp.asarray(ker)
        d_output = cp.zeros_like(d_input)

        block = (16, 16, 1)
        grid = ((w + block[0] - 1) // block[0], (h + block[1] - 1) // block[1], c)

        self._kernel(
            grid,
            block,
            (
                d_input,
                d_kernel,
                d_output,
                np.int32(h),
                np.int32(w),
                np.int32(c),
                np.int32(ksize),
                np.int32(pad),
            ),
        )
        cp.cuda.runtime.deviceSynchronize()

        out = cp.asnumpy(d_output)
        if image.ndim == 2:
            return out[:, :, 0]
        return out

    def benchmark_once_ms(self, image: np.ndarray, kernel: np.ndarray) -> float:
        cp = self.cp
        img = self._prepare_image(image)
        ker = self._prepare_kernel(kernel)

        h, w, c = img.shape
        ksize = int(ker.shape[0])
        pad = ksize // 2

        d_input = cp.asarray(img)
        d_kernel = cp.asarray(ker)
        d_output = cp.zeros_like(d_input)

        block = (16, 16, 1)
        grid = ((w + block[0] - 1) // block[0], (h + block[1] - 1) // block[1], c)

        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        self._kernel(
            grid,
            block,
            (
                d_input,
                d_kernel,
                d_output,
                np.int32(h),
                np.int32(w),
                np.int32(c),
                np.int32(ksize),
                np.int32(pad),
            ),
        )
        end.record()
        end.synchronize()
        return float(cp.cuda.get_elapsed_time(start, end))
