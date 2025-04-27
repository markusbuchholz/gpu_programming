#include <bits/stdc++.h>

__global__ void matrixAdd(const int *ma, const int *mb, int *m1, int N)
{
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Boundary check
    if (tid < N)
        m1[tid] = ma[tid] + mb[tid];
}


void init_matrix(int *matrix, int n)
{

    for (int ii = 0; ii < n; ii++)
    {
        for (int jj = 0; jj < n; jj++)
        {

            matrix[n * ii + jj] = rand() % 100;
        }
    }
}

int main()
{

    int N = 1 << 10;

    int *matrix_a = new int[N * N];
    int *matrix_b = new int[N * N];
    int *matrix_c = new int[N * N];
    int *matrix_d = new int[N * N];

    int *matrix_results = new int[N * N];

    init_matrix(matrix_a, N);
    init_matrix(matrix_b, N);
    init_matrix(matrix_c, N);
    init_matrix(matrix_d, N);

    size_t bytes_n = N * N * sizeof(int);

    int *ma, *mb, *mc, *md, *m1, *m2;

    // allocate memory on device (GPU)
    cudaMalloc(&ma, bytes_n);
    cudaMalloc(&mb, bytes_n);
    cudaMalloc(&mc, bytes_n);
    cudaMalloc(&md, bytes_n);
    cudaMalloc(&m1, bytes_n);
    cudaMalloc(&m2, bytes_n);

    // copy data from CPU to GPU

    cudaMemcpy(ma, matrix_a, bytes_n, cudaMemcpyHostToDevice);
    cudaMemcpy(mb, matrix_b, bytes_n, cudaMemcpyHostToDevice);
    cudaMemcpy(mc, matrix_c, bytes_n, cudaMemcpyHostToDevice);
    cudaMemcpy(md, matrix_d, bytes_n, cudaMemcpyHostToDevice);

    int NUM_THREADS = 256;
    int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

    // perform computaion on GPU
    matrixAdd<<<NUM_BLOCKS, NUM_THREADS>>>(ma, mb, m1, N*N);

    // Copy the result back to the CPU
    cudaMemcpy(matrix_results, m1, bytes_n, cudaMemcpyDeviceToHost);

    for (int ii = 0; ii < N; ii++)
    {

        for (int jj = 0; jj < N; jj++)
        {

            std::cout << matrix_results[N * ii + jj] << " ,";
        }
    }

    std::cout << " ------------------------------" << std::endl;

    // perform computaion on GPU
    matrixAdd<<<NUM_BLOCKS, NUM_THREADS>>>(mc, md, m2, N*N);

    // Copy the result back to the CPU
    cudaMemcpy(matrix_results, m2, bytes_n, cudaMemcpyDeviceToHost);

    for (int ii = 0; ii < N; ii++)
    {

        for (int jj = 0; jj < N; jj++)
        {

            std::cout << matrix_results[N * ii + jj] << " ,";
        }
    }

    
}