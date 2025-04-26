// gpu_matrix_multiplication_optimized.cu
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

// Error checking macro
#define cudaCheckErrors(msg)                                \
    do {                                                    \
        cudaError_t __err = cudaGetLastError();             \
        if (__err != cudaSuccess) {                         \
            std::cerr << "Fatal error: " << msg << std::endl;\
            std::cerr << "Error code: " << cudaGetErrorString(__err) << std::endl;\
            exit(1);                                        \
        }                                                   \
    } while (0)

// Matrix size (e.g., 1000x1000)
#define SIZE 5000

// Optimized matrix multiplication with shared memory
__global__ void matrixMultiplyShared(const float* A, const float* B, float* C, int N) {
    // Tile size
    const int TILE_SIZE = 16;

    // Shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Calculate row and column indices
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f;

    // Loop over tiles
    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        // Load tiles into shared memory
        if (row < N && (m * TILE_SIZE + threadIdx.x) < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + m * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (m * TILE_SIZE + threadIdx.y) < N)
            Bs[threadIdx.y][threadIdx.x] = B[(m * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < TILE_SIZE; ++k) {
            value += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result
    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

int main(int argc, char* argv[]) {
    int iterations = 1<<20; // Default number of iterations

    if (argc > 1) {
        iterations = atoi(argv[1]);
    }

    std::cout << "Matrix size: " << SIZE << "x" << SIZE << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;

    size_t bytes = SIZE * SIZE * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);

    // Initialize matrices with random values
    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < SIZE * SIZE; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, bytes);
    cudaCheckErrors("cudaMalloc d_A failed");
    cudaMalloc((void**)&d_B, bytes);
    cudaCheckErrors("cudaMalloc d_B failed");
    cudaMalloc((void**)&d_C, bytes);
    cudaCheckErrors("cudaMalloc d_C failed");

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy h_A to d_A failed");
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy h_B to d_B failed");

    // Define block and grid sizes
    const int TILE_SIZE = 16;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((SIZE + TILE_SIZE - 1) / TILE_SIZE, (SIZE + TILE_SIZE - 1) / TILE_SIZE);

    // Warm-up
    std::cout << "Warming up GPU..." << std::endl;
    matrixMultiplyShared<<<grid, block>>>(d_A, d_B, d_C, SIZE);
    cudaCheckErrors("Kernel launch failed");
    cudaDeviceSynchronize();
    std::cout << "Warm-up complete." << std::endl;

    // Start timing
    std::cout << "Starting matrix multiplication loop..." << std::endl;
    clock_t start = clock();

    for (int i = 0; i < iterations; ++i) {
        matrixMultiplyShared<<<grid, block>>>(d_A, d_B, d_C, SIZE);
        cudaCheckErrors("Kernel launch failed");
        // Optionally synchronize every 10 iterations
        if (i % 10 == 0) {
            cudaDeviceSynchronize();
            std::cout << "Iteration " << i << " completed." << std::endl;
        }
    }

    // Ensure all operations are complete
    cudaDeviceSynchronize();
    clock_t end = clock();

    double total_time = double(end - start) / CLOCKS_PER_SEC;
    std::cout << "Completed " << iterations << " iterations in " << total_time << " seconds." << std::endl;
    std::cout << "Average time per iteration: " << (total_time / iterations) << " seconds." << std::endl;

    // Optional: Copy result back to host (not necessary for computation)
    // cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    // cudaCheckErrors("cudaMemcpy d_C to h_C failed");

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
