extern "C" __global__ void conv2d_exact(
    const float* input,
    const float* kernel,
    float* output,
    int height,
    int width,
    int channels,
    int ksize,
    int pad
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;

    if (x >= width || y >= height || c >= channels) {
        return;
    }

    float acc = 0.0f;

    for (int ky = 0; ky < ksize; ++ky) {
        for (int kx = 0; kx < ksize; ++kx) {
            int iy = y + ky - pad;
            int ix = x + kx - pad;

            // Replicate border handling to match CPU reference implementation.
            if (iy < 0) {
                iy = 0;
            } else if (iy >= height) {
                iy = height - 1;
            }
            if (ix < 0) {
                ix = 0;
            } else if (ix >= width) {
                ix = width - 1;
            }

            int img_idx = (iy * width + ix) * channels + c;
            int ker_idx = ky * ksize + kx;
            acc += input[img_idx] * kernel[ker_idx];
        }
    }

    int out_idx = (y * width + x) * channels + c;
    output[out_idx] = acc;
}
