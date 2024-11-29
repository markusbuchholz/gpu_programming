#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>   
#include <ctime>     

// CUDA kernel for vector addition
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

int main() {
    srand(static_cast<unsigned>(time(0)));

    int N = 1 << 20; // 1000K elements
    size_t size = N * sizeof(float);

    // allocate host memory
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // allocate device memory
    float *d_A, *d_B, *d_C;
    cudaError_t err;

    err = cudaMalloc((void**)&d_A, size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for A (error code " 
                  << cudaGetErrorString(err) << ")!" << std::endl;
        return EXIT_FAILURE;
    }

    err = cudaMalloc((void**)&d_B, size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for B (error code " 
                  << cudaGetErrorString(err) << ")!" << std::endl;
        cudaFree(d_A);
        return EXIT_FAILURE;
    }

    err = cudaMalloc((void**)&d_C, size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for C (error code " 
                  << cudaGetErrorString(err) << ")!" << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        return EXIT_FAILURE;
    }

    // copy data from host to device
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy A from host to device (error code " 
                  << cudaGetErrorString(err) << ")!" << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return EXIT_FAILURE;
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy B from host to device (error code " 
                  << cudaGetErrorString(err) << ")!" << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return EXIT_FAILURE;
    }

    // launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // check for any errors launching the kernel
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch vectorAdd kernel (error code " 
                  << cudaGetErrorString(err) << ")!" << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return EXIT_FAILURE;
    }

    // copy result back to host
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy C from device to host (error code " 
                  << cudaGetErrorString(err) << ")!" << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return EXIT_FAILURE;
    }

    // Checks
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            std::cerr << "Mismatch at index " << i << ": " << h_C[i]
                      << " != " << h_A[i] + h_B[i] << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Vector addition successful!" << std::endl;
    } else {
        std::cout << "Vector addition failed!" << std::endl;
    }

    std::cout << "\nFirst 10 elements of each vector:" << std::endl;
    std::cout << "Index\tA\t\tB\t\tC = A + B" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << i << "\t" << h_A[i] << "\t" 
                  << h_B[i] << "\t" << h_C[i] << std::endl;
    }

    // free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}

