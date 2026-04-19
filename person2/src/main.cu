#include "io.hpp"

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                                        \
    do {                                                                                        \
        const cudaError_t error__ = (call);                                                     \
        if (error__ != cudaSuccess) {                                                           \
            std::ostringstream oss__;                                                           \
            oss__ << "CUDA error: " << cudaGetErrorString(error__);                             \
            throw std::runtime_error(oss__.str());                                              \
        }                                                                                       \
    } while (0)

#define CUSOLVER_CHECK(call)                                                                    \
    do {                                                                                        \
        const cusolverStatus_t status__ = (call);                                               \
        if (status__ != CUSOLVER_STATUS_SUCCESS) {                                              \
            std::ostringstream oss__;                                                           \
            oss__ << "cuSOLVER error code: " << static_cast<int>(status__);                     \
            throw std::runtime_error(oss__.str());                                              \
        }                                                                                       \
    } while (0)

#define MAX_RANK 32
#define MAX_RADIUS 16
#define FILTER_LENGTH (2 * MAX_RADIUS + 1)
#define TILE_W 16
#define TILE_H 16

__constant__ float c_RowWeights[MAX_RANK * FILTER_LENGTH];
__constant__ float c_ColWeights[MAX_RANK * FILTER_LENGTH];

namespace {

struct Options {
    std::string kernel_path;
    std::string kernel_name;
    int size = 0;
    std::vector<int> ranks;
    std::string output_dir;
    std::string image_path;
    int image_width = 0;
    int image_height = 0;
};

struct Metrics {
    int rank = 0;
    double frobenius_error = 0.0;
    double relative_frobenius_error = 0.0;
    double cumulative_energy = 0.0;
    double filter_time_ms = -1.0;
};

void print_usage() {
    std::cout
        << "Usage:\n"
        << "  Analysis mode:\n"
        << "    ./lowrank_svd --kernel_path outputs/disk_31.csv --kernel_name disk"
        << " --size 31 --ranks 1 2 4 8 --output_dir outputs/disk_31\n"
        << "  Fused filter mode:\n"
        << "    ./lowrank_svd --kernel_path outputs/disk_31.csv --kernel_name disk"
        << " --size 31 --ranks 1 2 4 8 --output_dir outputs/disk_31"
        << " --image_path images/input.csv --image_width 512 --image_height 512\n";
}

Options parse_args(int argc, char** argv) {
    Options options;
    options.output_dir = "outputs/run";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--kernel_path" && i + 1 < argc) {
            options.kernel_path = argv[++i];
        } else if (arg == "--kernel_name" && i + 1 < argc) {
            options.kernel_name = argv[++i];
        } else if (arg == "--size" && i + 1 < argc) {
            options.size = std::atoi(argv[++i]);
        } else if (arg == "--output_dir" && i + 1 < argc) {
            options.output_dir = argv[++i];
        } else if (arg == "--image_path" && i + 1 < argc) {
            options.image_path = argv[++i];
        } else if (arg == "--image_width" && i + 1 < argc) {
            options.image_width = std::atoi(argv[++i]);
        } else if (arg == "--image_height" && i + 1 < argc) {
            options.image_height = std::atoi(argv[++i]);
        } else if (arg == "--ranks") {
            ++i;
            while (i < argc && std::string(argv[i]).rfind("--", 0) != 0) {
                options.ranks.push_back(std::atoi(argv[i]));
                ++i;
            }
            --i;
        } else if (arg == "--help" || arg == "-h") {
            print_usage();
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + arg);
        }
    }

    if (options.kernel_path.empty()) {
        throw std::runtime_error("--kernel_path is required.");
    }
    if (options.kernel_name.empty()) {
        throw std::runtime_error("--kernel_name is required.");
    }
    if (options.size <= 0 || options.size % 2 == 0) {
        throw std::runtime_error("--size must be a positive odd integer.");
    }
    if (options.ranks.empty()) {
        throw std::runtime_error("--ranks requires at least one positive rank.");
    }

    std::sort(options.ranks.begin(), options.ranks.end());
    options.ranks.erase(std::unique(options.ranks.begin(), options.ranks.end()), options.ranks.end());

    for (std::size_t i = 0; i < options.ranks.size(); ++i) {
        if (options.ranks[i] <= 0 || options.ranks[i] > options.size) {
            throw std::runtime_error("Each rank must be between 1 and kernel size.");
        }
    }

    if (!options.image_path.empty()) {
        if (options.image_width <= 0 || options.image_height <= 0) {
            throw std::runtime_error(
                "--image_width and --image_height must be positive when --image_path is set.");
        }
    } else if (options.image_width > 0 || options.image_height > 0) {
        throw std::runtime_error("--image_path is required when image dimensions are provided.");
    }

    return options;
}

bool fused_mode_enabled(const Options& options) {
    return !options.image_path.empty();
}

std::vector<double> row_major_to_col_major(const std::vector<double>& row_major, int rows, int cols) {
    std::vector<double> col_major(static_cast<std::size_t>(rows) * cols, 0.0);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            col_major[static_cast<std::size_t>(col) * rows + row] =
                row_major[static_cast<std::size_t>(row) * cols + col];
        }
    }
    return col_major;
}

std::vector<double> extract_row_vectors(const std::vector<double>& vt_col_major, int size, int rank) {
    std::vector<double> rows(static_cast<std::size_t>(rank) * size, 0.0);
    for (int component = 0; component < rank; ++component) {
        for (int col = 0; col < size; ++col) {
            rows[static_cast<std::size_t>(component) * size + col] =
                vt_col_major[component + static_cast<std::size_t>(col) * size];
        }
    }
    return rows;
}

std::vector<double> extract_column_vectors(
    const std::vector<double>& u_col_major,
    const std::vector<double>& singular_values,
    int size,
    int rank) {
    std::vector<double> cols(static_cast<std::size_t>(rank) * size, 0.0);
    for (int component = 0; component < rank; ++component) {
        for (int row = 0; row < size; ++row) {
            cols[static_cast<std::size_t>(component) * size + row] =
                singular_values[component] * u_col_major[static_cast<std::size_t>(component) * size + row];
        }
    }
    return cols;
}

std::vector<float> to_float_vector(const std::vector<double>& input) {
    std::vector<float> output(input.size(), 0.0f);
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = static_cast<float>(input[i]);
    }
    return output;
}

std::vector<double> to_double_vector(const std::vector<float>& input) {
    std::vector<double> output(input.size(), 0.0);
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = static_cast<double>(input[i]);
    }
    return output;
}

double frobenius_norm(const std::vector<double>& values) {
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        sum_sq += values[i] * values[i];
    }
    return std::sqrt(sum_sq);
}

double frobenius_error(const std::vector<double>& a, const std::vector<double>& b) {
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double diff = a[i] - b[i];
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq);
}

double cumulative_energy(const std::vector<double>& singular_values, int rank) {
    double kept = 0.0;
    double total = 0.0;
    for (std::size_t i = 0; i < singular_values.size(); ++i) {
        const double term = singular_values[i] * singular_values[i];
        total += term;
        if (static_cast<int>(i) < rank) {
            kept += term;
        }
    }
    return total > 0.0 ? kept / total : 0.0;
}

std::string rank_prefix(const std::string& output_dir, int rank) {
    std::ostringstream oss;
    oss << join_path(output_dir, "rank_" + std::to_string(rank));
    return oss.str();
}

void validate_fused_filter_limits(int rank, int radius) {
    if (rank > MAX_RANK) {
        std::ostringstream oss;
        oss << "Fused filter rank " << rank << " exceeds MAX_RANK=" << MAX_RANK
            << " from person3's original CUDA implementation.";
        throw std::runtime_error(oss.str());
    }
    if (radius > MAX_RADIUS) {
        std::ostringstream oss;
        oss << "Kernel radius " << radius << " exceeds MAX_RADIUS=" << MAX_RADIUS
            << " from person3's original CUDA implementation.";
        throw std::runtime_error(oss.str());
    }
}

std::string build_rank_log(
    const Options& options,
    const Metrics& metrics,
    const std::string& reconstruction_path,
    const std::string& row_vectors_path,
    const std::string& column_vectors_path,
    const std::string& filtered_output_path) {
    std::ostringstream oss;
    oss << std::setprecision(17);
    oss << "kernel_name=" << options.kernel_name << '\n';
    oss << "kernel_path=" << options.kernel_path << '\n';
    oss << "size=" << options.size << '\n';
    oss << "rank=" << metrics.rank << '\n';
    oss << "frobenius_error=" << metrics.frobenius_error << '\n';
    oss << "relative_frobenius_error=" << metrics.relative_frobenius_error << '\n';
    oss << "cumulative_energy_retained=" << metrics.cumulative_energy << '\n';
    if (!reconstruction_path.empty()) {
        oss << "reconstruction_csv=" << reconstruction_path << '\n';
    }
    if (!row_vectors_path.empty()) {
        oss << "row_vectors_csv=" << row_vectors_path << '\n';
    }
    if (!column_vectors_path.empty()) {
        oss << "column_vectors_csv=" << column_vectors_path << '\n';
    }
    if (!filtered_output_path.empty()) {
        oss << "input_image_csv=" << options.image_path << '\n';
        oss << "image_width=" << options.image_width << '\n';
        oss << "image_height=" << options.image_height << '\n';
        oss << "filtered_output_csv=" << filtered_output_path << '\n';
        oss << "filter_time_ms=" << metrics.filter_time_ms << '\n';
    }
    return oss.str();
}

__global__ void fusedRowKernel(
    float* d_Intermediate,
    const float* d_Input,
    int width,
    int height,
    int rank,
    int radius) {
    __shared__ float s_Data[TILE_H][TILE_W + 2 * MAX_RADIUS];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_W + tx;
    int y = blockIdx.y * TILE_H + ty;

    int clamped_x = min(max(x, 0), width - 1);
    int clamped_y = min(max(y, 0), height - 1);
    s_Data[ty][tx + radius] = d_Input[clamped_y * width + clamped_x];

    if (tx < radius) {
        int left_x = min(max(x - radius, 0), width - 1);
        s_Data[ty][tx] = d_Input[clamped_y * width + left_x];
    }
    if (tx >= TILE_W - radius) {
        int right_x = min(max(x + radius, 0), width - 1);
        s_Data[ty][tx + 2 * radius] = d_Input[clamped_y * width + right_x];
    }

    __syncthreads();

    if (x < width && y < height) {
        for (int r = 0; r < rank; r++) {
            float sum = 0.0f;
            int weight_offset = r * (2 * radius + 1);

            #pragma unroll
            for (int i = -radius; i <= radius; i++) {
                sum += s_Data[ty][tx + radius + i] * c_RowWeights[weight_offset + (i + radius)];
            }

            int out_idx = (r * width * height) + (y * width) + x;
            d_Intermediate[out_idx] = sum;
        }
    }
}

__global__ void fusedColAndAccumulateKernel(
    float* d_Output,
    const float* d_Intermediate,
    int width,
    int height,
    int rank,
    int radius) {
    __shared__ float s_Data[TILE_W][TILE_H + 2 * MAX_RADIUS];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_W + tx;
    int y = blockIdx.y * TILE_H + ty;

    float final_pixel_sum = 0.0f;

    for (int r = 0; r < rank; r++) {
        int channel_offset = r * width * height;

        int clamped_x = min(max(x, 0), width - 1);
        int clamped_y = min(max(y, 0), height - 1);
        s_Data[tx][ty + radius] = d_Intermediate[channel_offset + clamped_y * width + clamped_x];

        if (ty < radius) {
            int top_y = min(max(y - radius, 0), height - 1);
            s_Data[tx][ty] = d_Intermediate[channel_offset + top_y * width + clamped_x];
        }
        if (ty >= TILE_H - radius) {
            int bottom_y = min(max(y + radius, 0), height - 1);
            s_Data[tx][ty + 2 * radius] = d_Intermediate[channel_offset + bottom_y * width + clamped_x];
        }

        __syncthreads();

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

    if (x < width && y < height) {
        d_Output[y * width + x] = final_pixel_sum;
    }
}

float runFusedLowRankFilter(
    const float* h_InputImage,
    float* h_OutputImage,
    int width,
    int height,
    int rank,
    int radius,
    const float* h_RowWeights,
    const float* h_ColWeights) {
    validate_fused_filter_limits(rank, radius);

    int filter_length = 2 * radius + 1;
    size_t image_size = static_cast<size_t>(width) * height * sizeof(float);
    size_t intermediate_size = static_cast<size_t>(rank) * image_size;

    CUDA_CHECK(cudaMemcpyToSymbol(c_RowWeights, h_RowWeights, rank * filter_length * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_ColWeights, h_ColWeights, rank * filter_length * sizeof(float)));

    float* d_Input = nullptr;
    float* d_Intermediate = nullptr;
    float* d_Output = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    try {
        CUDA_CHECK(cudaMalloc(&d_Input, image_size));
        CUDA_CHECK(cudaMalloc(&d_Intermediate, intermediate_size));
        CUDA_CHECK(cudaMalloc(&d_Output, image_size));
        CUDA_CHECK(cudaMemcpy(d_Input, h_InputImage, image_size, cudaMemcpyHostToDevice));

        dim3 blockDim(TILE_W, TILE_H);
        dim3 gridDim((width + TILE_W - 1) / TILE_W, (height + TILE_H - 1) / TILE_H);

        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));

        fusedRowKernel<<<gridDim, blockDim>>>(d_Intermediate, d_Input, width, height, rank, radius);
        CUDA_CHECK(cudaGetLastError());
        fusedColAndAccumulateKernel<<<gridDim, blockDim>>>(d_Output, d_Intermediate, width, height, rank, radius);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        CUDA_CHECK(cudaMemcpy(h_OutputImage, d_Output, image_size, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_Input));
        CUDA_CHECK(cudaFree(d_Intermediate));
        CUDA_CHECK(cudaFree(d_Output));
        return milliseconds;
    } catch (...) {
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
        if (d_Input != nullptr) {
            cudaFree(d_Input);
        }
        if (d_Intermediate != nullptr) {
            cudaFree(d_Intermediate);
        }
        if (d_Output != nullptr) {
            cudaFree(d_Output);
        }
        throw;
    }
}

__global__ void reconstruct_rank_kernel(
    const double* u_col_major,
    const double* singular_values,
    const double* vt_col_major,
    int size,
    int rank,
    double* reconstruction_row_major) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= size || col >= size) {
        return;
    }

    double value = 0.0;
    for (int component = 0; component < rank; ++component) {
        value += singular_values[component]
            * u_col_major[static_cast<std::size_t>(component) * size + row]
            * vt_col_major[component + static_cast<std::size_t>(col) * size];
    }

    reconstruction_row_major[static_cast<std::size_t>(row) * size + col] = value;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const bool fused_mode = fused_mode_enabled(options);
        const std::vector<double> kernel_row_major =
            load_csv_matrix(options.kernel_path, options.size, options.size);
        const std::vector<double> kernel_col_major =
            row_major_to_col_major(kernel_row_major, options.size, options.size);
        const double kernel_frobenius = frobenius_norm(kernel_row_major);

        std::vector<float> input_image;
        if (fused_mode) {
            input_image = to_float_vector(
                load_csv_matrix(options.image_path, options.image_height, options.image_width));
        }

        cusolverDnHandle_t solver_handle = nullptr;
        CUSOLVER_CHECK(cusolverDnCreate(&solver_handle));

        double* d_kernel = nullptr;
        double* d_singular_values = nullptr;
        double* d_u = nullptr;
        double* d_vt = nullptr;
        double* d_work = nullptr;
        double* d_reconstruction = nullptr;
        int* d_info = nullptr;

        try {
            const int matrix_elements = options.size * options.size;
            const int singular_count = options.size;

            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_kernel), sizeof(double) * matrix_elements));
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_singular_values),
                sizeof(double) * singular_count));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_u), sizeof(double) * matrix_elements));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vt), sizeof(double) * matrix_elements));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int)));
            CUDA_CHECK(cudaMemcpy(
                d_kernel,
                kernel_col_major.data(),
                sizeof(double) * matrix_elements,
                cudaMemcpyHostToDevice));

            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(
                solver_handle,
                options.size,
                options.size,
                &lwork));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_work), sizeof(double) * lwork));

            CUSOLVER_CHECK(cusolverDnDgesvd(
                solver_handle,
                'A',
                'A',
                options.size,
                options.size,
                d_kernel,
                options.size,
                d_singular_values,
                d_u,
                options.size,
                d_vt,
                options.size,
                d_work,
                lwork,
                nullptr,
                d_info));

            int info = 0;
            CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
            if (info != 0) {
                std::ostringstream oss;
                oss << "cuSOLVER gesvd returned info=" << info;
                throw std::runtime_error(oss.str());
            }

            std::vector<double> singular_values(singular_count, 0.0);
            std::vector<double> u_col_major(matrix_elements, 0.0);
            std::vector<double> vt_col_major(matrix_elements, 0.0);

            CUDA_CHECK(cudaMemcpy(
                singular_values.data(),
                d_singular_values,
                sizeof(double) * singular_count,
                cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(
                u_col_major.data(),
                d_u,
                sizeof(double) * matrix_elements,
                cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(
                vt_col_major.data(),
                d_vt,
                sizeof(double) * matrix_elements,
                cudaMemcpyDeviceToHost));

            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&d_reconstruction),
                sizeof(double) * matrix_elements));

            if (!fused_mode) {
                write_csv_matrix(
                    join_path(options.output_dir, "input_kernel.csv"),
                    kernel_row_major,
                    options.size,
                    options.size);
                write_csv_matrix(
                    join_path(options.output_dir, "singular_values.csv"),
                    singular_values,
                    1,
                    singular_count);
            }

            std::ostringstream metrics_csv;
            metrics_csv << "rank,frobenius_error,relative_frobenius_error,cumulative_energy_retained,filter_time_ms\n";

            std::ostringstream run_log;
            run_log << std::setprecision(17);
            run_log << "mode=" << (fused_mode ? "fused_filter" : "analysis") << '\n';
            run_log << "kernel_name=" << options.kernel_name << '\n';
            run_log << "kernel_path=" << options.kernel_path << '\n';
            run_log << "size=" << options.size << '\n';
            run_log << "output_dir=" << options.output_dir << '\n';
            run_log << "kernel_frobenius_norm=" << kernel_frobenius << '\n';
            if (fused_mode) {
                run_log << "image_path=" << options.image_path << '\n';
                run_log << "image_width=" << options.image_width << '\n';
                run_log << "image_height=" << options.image_height << '\n';
            }
            run_log << "ranks=";
            for (std::size_t i = 0; i < options.ranks.size(); ++i) {
                if (i > 0) {
                    run_log << ' ';
                }
                run_log << options.ranks[i];
            }
            run_log << '\n';

            std::cout << std::left << std::setw(8) << "rank"
                      << std::setw(20) << "fro_error"
                      << std::setw(20) << "rel_fro_error"
                      << std::setw(20) << "energy_retained"
                      << std::setw(16) << "filter_ms" << '\n';

            const dim3 block(16, 16);
            const dim3 grid(
                static_cast<unsigned int>((options.size + block.x - 1) / block.x),
                static_cast<unsigned int>((options.size + block.y - 1) / block.y));

            for (std::size_t i = 0; i < options.ranks.size(); ++i) {
                const int rank = options.ranks[i];
                reconstruct_rank_kernel<<<grid, block>>>(
                    d_u,
                    d_singular_values,
                    d_vt,
                    options.size,
                    rank,
                    d_reconstruction);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());

                std::vector<double> reconstruction(matrix_elements, 0.0);
                CUDA_CHECK(cudaMemcpy(
                    reconstruction.data(),
                    d_reconstruction,
                    sizeof(double) * matrix_elements,
                    cudaMemcpyDeviceToHost));

                Metrics metrics;
                metrics.rank = rank;
                metrics.frobenius_error = frobenius_error(kernel_row_major, reconstruction);
                metrics.relative_frobenius_error =
                    kernel_frobenius > 0.0 ? metrics.frobenius_error / kernel_frobenius : 0.0;
                metrics.cumulative_energy = cumulative_energy(singular_values, rank);

                const std::string prefix = rank_prefix(options.output_dir, rank);
                const std::string reconstruction_path =
                    fused_mode ? std::string() : prefix + "_reconstruction.csv";
                const std::string row_vectors_path =
                    fused_mode ? std::string() : prefix + "_rows.csv";
                const std::string column_vectors_path =
                    fused_mode ? std::string() : prefix + "_cols.csv";
                const std::string filtered_output_path =
                    fused_mode ? prefix + "_filtered.csv" : std::string();
                const std::string log_path = prefix + "_log.txt";

                if (fused_mode) {
                    validate_fused_filter_limits(rank, options.size / 2);
                    const std::vector<float> row_weights =
                        to_float_vector(extract_row_vectors(vt_col_major, options.size, rank));
                    const std::vector<float> col_weights =
                        to_float_vector(extract_column_vectors(u_col_major, singular_values, options.size, rank));
                    std::vector<float> filtered_image(
                        static_cast<std::size_t>(options.image_width) * options.image_height,
                        0.0f);

                    metrics.filter_time_ms = runFusedLowRankFilter(
                        input_image.data(),
                        filtered_image.data(),
                        options.image_width,
                        options.image_height,
                        rank,
                        options.size / 2,
                        row_weights.data(),
                        col_weights.data());

                    write_csv_matrix(
                        filtered_output_path,
                        to_double_vector(filtered_image),
                        options.image_height,
                        options.image_width);
                } else {
                    write_csv_matrix(reconstruction_path, reconstruction, options.size, options.size);
                    write_csv_matrix(
                        row_vectors_path,
                        extract_row_vectors(vt_col_major, options.size, rank),
                        rank,
                        options.size);
                    write_csv_matrix(
                        column_vectors_path,
                        extract_column_vectors(u_col_major, singular_values, options.size, rank),
                        rank,
                        options.size);
                }

                write_text_file(
                    log_path,
                    build_rank_log(
                        options,
                        metrics,
                        reconstruction_path,
                        row_vectors_path,
                        column_vectors_path,
                        filtered_output_path));

                metrics_csv << std::setprecision(17)
                            << metrics.rank << ','
                            << metrics.frobenius_error << ','
                            << metrics.relative_frobenius_error << ','
                            << metrics.cumulative_energy << ','
                            << metrics.filter_time_ms << '\n';

                std::cout << std::left << std::setw(8) << metrics.rank
                          << std::setw(20) << std::setprecision(10) << metrics.frobenius_error
                          << std::setw(20) << metrics.relative_frobenius_error
                          << std::setw(20) << metrics.cumulative_energy
                          << std::setw(16) << metrics.filter_time_ms << '\n';
            }

            write_text_file(join_path(options.output_dir, "metrics.csv"), metrics_csv.str());
            write_text_file(join_path(options.output_dir, "run_log.txt"), run_log.str());

            CUDA_CHECK(cudaFree(d_reconstruction));
            CUDA_CHECK(cudaFree(d_work));
            CUDA_CHECK(cudaFree(d_info));
            CUDA_CHECK(cudaFree(d_vt));
            CUDA_CHECK(cudaFree(d_u));
            CUDA_CHECK(cudaFree(d_singular_values));
            CUDA_CHECK(cudaFree(d_kernel));
            CUSOLVER_CHECK(cusolverDnDestroy(solver_handle));
            return 0;
        } catch (...) {
            if (d_reconstruction != nullptr) {
                cudaFree(d_reconstruction);
            }
            if (d_work != nullptr) {
                cudaFree(d_work);
            }
            if (d_info != nullptr) {
                cudaFree(d_info);
            }
            if (d_vt != nullptr) {
                cudaFree(d_vt);
            }
            if (d_u != nullptr) {
                cudaFree(d_u);
            }
            if (d_singular_values != nullptr) {
                cudaFree(d_singular_values);
            }
            if (d_kernel != nullptr) {
                cudaFree(d_kernel);
            }
            if (solver_handle != nullptr) {
                cusolverDnDestroy(solver_handle);
            }
            throw;
        }
    } catch (const std::exception& ex) {
        std::cerr << "lowrank_svd failed: " << ex.what() << '\n';
        print_usage();
        return 1;
    }
}
