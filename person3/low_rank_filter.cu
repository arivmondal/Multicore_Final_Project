#include <iostream>
#include <cuda_runtime.h>

#define MAX_RANK 10 // Adjust based maximum SVD rank
#define MAX_RADIUS 16
#define FILTER_LENGTH (2 * MAX_RADIUS + 1)
#define TILE_W 16
#define TILE_H 16

// --- Constant Memory for SVD Weights ---
// Indexing: rank_index * FILTER_LENGTH + weight_index
__constant__ float c_RowWeights[MAX_RANK * FILTER_LENGTH];
__constant__ float c_ColWeights[MAX_RANK * FILTER_LENGTH];


// row kernel
// Reads 2D image once. Computes 'k' row blurs. Writes to 3D intermediate.
__global__ void fusedRowKernel(
    float* d_Intermediate, 
    const float* d_Input, 
    int width, 
    int height, 
    int rank, 
    int radius) 
{
    // Shared memory for a single row tile, including halo (ghost) cells
    __shared__ float s_Data[TILE_H][TILE_W + 2 * MAX_RADIUS];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_W + tx;
    int y = blockIdx.y * TILE_H + ty;

    // Load Main Tile into Shared Memory (with clamping for edges)
    int clamped_x = min(max(x, 0), width - 1);
    int clamped_y = min(max(y, 0), height - 1);
    s_Data[ty][tx + radius] = d_Input[clamped_y * width + clamped_x];

    // Load Left Halo
    if (tx < radius) {
        int left_x = min(max(x - radius, 0), width - 1);
        s_Data[ty][tx] = d_Input[clamped_y * width + left_x];
    }
    // Load Right Halo
    if (tx >= TILE_W - radius) {
        int right_x = min(max(x + radius, 0), width - 1);
        s_Data[ty][tx + 2 * radius] = d_Input[clamped_y * width + right_x];
    }

    __syncthreads();

    // Compute Convolution for all K ranks
    if (x < width && y < height) {
        for (int r = 0; r < rank; r++) {
            float sum = 0.0f;
            int weight_offset = r * (2 * radius + 1);
            
            #pragma unroll
            for (int i = -radius; i <= radius; i++) {
                sum += s_Data[ty][tx + radius + i] * c_RowWeights[weight_offset + (i + radius)];
            }
            
            // Write to 3D intermediate buffer (Channel/Rank -> Y -> X)
            int out_idx = (r * width * height) + (y * width) + x;
            d_Intermediate[out_idx] = sum;
        }
    }
}


// column kernel
// Reads 3D intermediate. Computes 'k' col blurs. Accumulates in register.
__global__ void fusedColAndAccumulateKernel(
    float* d_Output, 
    const float* d_Intermediate, 
    int width, 
    int height, 
    int rank, 
    int radius) 
{
    // Shared memory for column tile (Note: transposed shape for memory coalescing)
    __shared__ float s_Data[TILE_W][TILE_H + 2 * MAX_RADIUS];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_W + tx;
    int y = blockIdx.y * TILE_H + ty;

    // Register to accumulate the final pixel value across all ranks
    float final_pixel_sum = 0.0f;

    // Loop over each rank, reusing the same shared memory to save space
    for (int r = 0; r < rank; r++) {
        
        int channel_offset = r * width * height;

        // 1. Load Main Tile for current rank
        int clamped_x = min(max(x, 0), width - 1);
        int clamped_y = min(max(y, 0), height - 1);
        s_Data[tx][ty + radius] = d_Intermediate[channel_offset + clamped_y * width + clamped_x];

        // 2. Load Top Halo
        if (ty < radius) {
            int top_y = min(max(y - radius, 0), height - 1);
            s_Data[tx][ty] = d_Intermediate[channel_offset + top_y * width + clamped_x];
        }
        // 3. Load Bottom Halo
        if (ty >= TILE_H - radius) {
            int bottom_y = min(max(y + radius, 0), height - 1);
            s_Data[tx][ty + 2 * radius] = d_Intermediate[channel_offset + bottom_y * width + clamped_x];
        }

        __syncthreads();

        // Compute column convolution and accumulate in register
        if (x < width && y < height) {
            float col_sum = 0.0f;
            int weight_offset = r * (2 * radius + 1);

            #pragma unroll
            for (int i = -radius; i <= radius; i++) {
                col_sum += s_Data[tx][ty + radius + i] * c_ColWeights[weight_offset + (i + radius)];
            }
            
            final_pixel_sum += col_sum; 
        }
        
        __syncthreads();
    }

    // Write final accumulated result exactly ONCE to global memory
    if (x < width && y < height) {
        d_Output[y * width + x] = final_pixel_sum;
    }
}
// host
float runFusedLowRankFilter(
    const float* h_InputImage,   // Host pointer to original image (1D flat grayscale array)
    float* h_OutputImage,        // Host pointer for final image
    int width,                   
    int height,                  
    int rank,                    // 'k' provided by Person 2
    int radius,                  // filter radius (size = 2*radius + 1)
    const float* h_RowWeights,   // Host array of size (rank * filter_length)
    const float* h_ColWeights)   // Host array of size (rank * filter_length)
{
    int filter_length = 2 * radius + 1;
    size_t image_size = width * height * sizeof(float);
    size_t intermediate_size = rank * image_size;

    // Copy weights to Constant Memory
    cudaMemcpyToSymbol(c_RowWeights, h_RowWeights, rank * filter_length * sizeof(float));
    cudaMemcpyToSymbol(c_ColWeights, h_ColWeights, rank * filter_length * sizeof(float));

    // Allocate Device Memory
    float *d_Input, *d_Intermediate, *d_Output;
    cudaMalloc(&d_Input, image_size);
    cudaMalloc(&d_Intermediate, intermediate_size);
    cudaMalloc(&d_Output, image_size);

    // Copy Input Image to Device
    cudaMemcpy(d_Input, h_InputImage, image_size, cudaMemcpyHostToDevice);

    // Setup Grid and Block Dimensions
    dim3 blockDim(TILE_W, TILE_H);
    dim3 gridDim((width + TILE_W - 1) / TILE_W, (height + TILE_H - 1) / TILE_H);

    // Setup CUDA Events for highly accurate timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // fusedRowKernel<<<gridDim, blockDim>>>(d_Intermediate, d_Input, width, height, rank, radius);
    // cudaDeviceSynchronize();

    // start
    cudaEventRecord(start);

    fusedRowKernel<<<gridDim, blockDim>>>(d_Intermediate, d_Input, width, height, rank, radius);
    fusedColAndAccumulateKernel<<<gridDim, blockDim>>>(d_Output, d_Intermediate, width, height, rank, radius);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    //end

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy final image back to Host
    cudaMemcpy(h_OutputImage, d_Output, image_size, cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_Input);
    cudaFree(d_Intermediate);
    cudaFree(d_Output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

int main() {
    // dummy dim
    int width = 32;
    int height = 32;
    int rank = 1;
    int radius = 1;
    int filter_length = 2 * radius + 1;

    size_t num_pixels = width * height;
    
    float* h_Input = new float[num_pixels];
    float* h_Output = new float[num_pixels];
    float* h_RowWeights = new float[rank * filter_length];
    float* h_ColWeights = new float[rank * filter_length];

    // dummy white image
    for(int i = 0; i < num_pixels; i++) {
        h_Input[i] = 1.0f;
    }
    
    // Weights: A simple 1/3 average blur for rank 0
    for(int i = 0; i < filter_length; i++) {
        h_RowWeights[i] = 1.0f / 3.0f;
        h_ColWeights[i] = 1.0f / 3.0f;
    }

    // Run the GPU Pipeline
    std::cout << "Starting GPU Filter..." << std::endl;
    float time_ms = runFusedLowRankFilter(h_Input, h_Output, width, height, rank, radius, h_RowWeights, h_ColWeights);

    // Verify Output (Quick check of the center pixel)
    int center_idx = (height / 2) * width + (width / 2);
    std::cout << "GPU Execution Time: " << time_ms << " ms" << std::endl;
    std::cout << "Center pixel value (Expected ~1.0): " << h_Output[center_idx] << std::endl;

    // 6. Cleanup
    delete[] h_Input;
    delete[] h_Output;
    delete[] h_RowWeights;
    delete[] h_ColWeights;

    return 0;
}
// Helper function to check if two floats are basically equal
// bool isClose(float a, float b, float epsilon = 1e-4) {
//     return std::abs(a - b) < epsilon;
// }

// void testIdentityFilter() {
//     std::cout << "[TEST 1] The Identity Filter (Math Correctness)" << std::endl;
//     // An identity filter should return the EXACT same image you put in.
//     // This proves your X/Y indexing and accumulation logic is perfectly aligned.
    
//     int w = 64, h = 64, rank = 1, radius = 2;
//     int filter_len = 2 * radius + 1;
//     float *in = new float[w * h], *out = new float[w * h];
//     float *r_weights = new float[rank * filter_len](); // Initializes to 0
//     float *c_weights = new float[rank * filter_len]();

//     // Fill image with sequential numbers
//     for (int i = 0; i < w * h; i++) in[i] = (float)i;

//     // Set ONLY the center weight to 1.0
//     r_weights[radius] = 1.0f;
//     c_weights[radius] = 1.0f;

//     runFusedLowRankFilter(in, out, w, h, rank, radius, r_weights, c_weights);

//     // Verify output matches input perfectly
//     bool pass = true;
//     for (int i = 0; i < w * h; i++) {
//         if (!isClose(in[i], out[i])) { pass = false; break; }
//     }
//     std::cout << (pass ? "  -> PASS\n" : "  -> FAIL (Data mismatch)\n");
    
//     delete[] in; delete[] out; delete[] r_weights; delete[] c_weights;
// }

// void testOddDimensions() {
//     std::cout << "[TEST 2] Asymmetric & Prime Dimensions (Bounds Checking)" << std::endl;
//     // Tests if bounds checking (x < width && y < height) works when 
//     // the image doesn't divide evenly into 16x16 tiles.
    
//     int w = 511; // Prime-ish, not divisible by 16
//     int h = 333; // Prime-ish, not divisible by 16
//     int rank = 1, radius = 1;
//     float *in = new float[w * h](), *out = new float[w * h]();
//     float r_weights[] = {0, 1, 0};
//     float c_weights[] = {0, 1, 0};

//     // If bounds checking fails, this will Segfault or throw a CUDA Error.
//     runFusedLowRankFilter(in, out, w, h, rank, radius, r_weights, c_weights);
    
//     // Check for standard CUDA errors after async execution
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         std::cout << "  -> FAIL: " << cudaGetErrorString(err) << "\n";
//     } else {
//         std::cout << "  -> PASS (No segfaults or out-of-bounds writes)\n";
//     }

//     delete[] in; delete[] out;
// }

// void testCornerHalo() {
//     std::cout << "[TEST 3] Corner Pixel Clamp-to-Edge (Shared Memory Logic)" << std::endl;
//     // Put a single bright pixel in the top-left corner.
//     // An average blur should pull that brightness into the surrounding pixels.
    
//     int w = 32, h = 32, rank = 1, radius = 1;
//     float *in = new float[w * h](), *out = new float[w * h]();
//     float r_weights[] = {0.5, 0.5, 0}; // Blur right
//     float c_weights[] = {0.5, 0.5, 0}; // Blur down

//     in[0] = 100.0f; // Spike at Top-Left corner (x=0, y=0)

//     runFusedLowRankFilter(in, out, w, h, rank, radius, r_weights, c_weights);

//     // The pixel at (0,0) should be clamped and blurred into itself and (1,1)
//     if (out[0] > 0.0f && out[1] > 0.0f) {
//         std::cout << "  -> PASS\n";
//     } else {
//         std::cout << "  -> FAIL (Halo logic at corner failed)\n";
//     }

//     delete[] in; delete[] out;
// }

// void test4KStress() {
//     std::cout << "[TEST 4] 4K Image Stress Test (Performance & OOM)" << std::endl;
//     // Simulates a massive real-world workload.
    
//     int w = 3840, h = 2160; // 4K Resolution
//     int rank = MAX_RANK;    // Max out the __constant__ memory and accumulation loop
//     int radius = MAX_RADIUS;
//     int filter_len = 2 * radius + 1;
    
//     size_t num_pixels = (size_t)w * h;
//     float *in = new float[num_pixels]();
//     float *out = new float[num_pixels]();
//     float *r_weights = new float[rank * filter_len]();
//     float *c_weights = new float[rank * filter_len]();

//     std::cout << "  -> Processing " << w << "x" << h << " with Rank " << rank << "..." << std::endl;
    
//     float time = runFusedLowRankFilter(in, out, w, h, rank, radius, r_weights, c_weights);
    
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         std::cout << "  -> FAIL: " << cudaGetErrorString(err) << "\n";
//     } else {
//         std::cout << "  -> PASS\n";
//         std::cout << "  -> GPU Time: " << time << " ms\n";
//     }

//     delete[] in; delete[] out; delete[] r_weights; delete[] c_weights;
// }

// int main() {
//     std::cout << "======================================\n";
//     std::cout << " RUNNING CUDA ENGINE EXHAUSTIVE TESTS \n";
//     std::cout << "======================================\n\n";

//     testIdentityFilter();
//     std::cout << "\n";
    
//     testOddDimensions();
//     std::cout << "\n";

//     testCornerHalo();
//     std::cout << "\n";

//     test4KStress();
//     std::cout << "\n======================================\n";

//     return 0;
// }