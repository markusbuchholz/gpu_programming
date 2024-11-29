// sobel fileter

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem> 
#include <cstring>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace fs = std::filesystem;

// CUDA kernel 
__global__ void rgb_to_grayscale(unsigned char* d_input, unsigned char* d_grayscale, int width, int height, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    if (idx < total_pixels) {
        int input_idx = idx * channels;
        unsigned char r = d_input[input_idx];
        unsigned char g = d_input[input_idx + 1];
        unsigned char b = d_input[input_idx + 2];

        unsigned char gray = static_cast<unsigned char>(0.21f * r + 0.72f * g + 0.07f * b);

        d_grayscale[idx] = gray;
    }
}

// CUDA kernel to apply the Sobel filter for edge detection
__global__ void sobel_filter(unsigned char* d_grayscale, unsigned char* d_edges, int width, int height) {
    // calculate the row and column index of the pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Column
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Row

    // ensure the pixel is not on the border
    if (x > 0 && x < (width - 1) && y > 0 && y < (height - 1)) {
        // compute the 1D index of the current pixel
        int idx = y * width + x;

        // def Sobel kernels
        int Gx = -1 * d_grayscale[(y - 1) * width + (x - 1)] + 1 * d_grayscale[(y - 1) * width + (x + 1)]
                 -2 * d_grayscale[y * width + (x - 1)]     + 2 * d_grayscale[y * width + (x + 1)]
                 -1 * d_grayscale[(y + 1) * width + (x - 1)] + 1 * d_grayscale[(y + 1) * width + (x + 1)];

        int Gy = -1 * d_grayscale[(y - 1) * width + (x - 1)] -2 * d_grayscale[(y - 1) * width + x] -1 * d_grayscale[(y - 1) * width + (x + 1)]
                 +1 * d_grayscale[(y + 1) * width + (x - 1)] +2 * d_grayscale[(y + 1) * width + x] +1 * d_grayscale[(y + 1) * width + (x + 1)];

        // compute the gradient magnitude
        int magnitude = abs(Gx) + abs(Gy);

        // clamp the result to [0, 255]
        magnitude = magnitude > 255 ? 255 : magnitude;

        d_edges[idx] = static_cast<unsigned char>(magnitude);
    }
    else if (x < width && y < height) {
        // For border pixels, set edge value to 0
        int idx = y * width + x;
        if (x >= width || y >= height) return; // Out of bounds
        d_edges[idx] = 0;
    }
}

int main() {

    std::string input_file = "../data/preikestolen.png";        
    std::string output_file_gray = "../data/preikestolen_grey.png";      
    std::string output_file_edges = "../data/preikestolen_edges.png";    

    if (!fs::exists(input_file) || !fs::is_regular_file(input_file)) {
        std::cerr << "Input file does not exist or is not a regular file." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Processing: " << fs::path(input_file).filename() << std::endl;

    int width, height, channels;

    unsigned char* h_input = stbi_load(input_file.c_str(), &width, &height, &channels, 3);
    if (h_input == nullptr) {
        std::cerr << "Failed to load image: " << input_file << std::endl;
        return EXIT_FAILURE;
    }

    channels = 3; 

    size_t input_size = width * height * channels * sizeof(unsigned char);
    size_t grayscale_size = width * height * sizeof(unsigned char);
    size_t edges_size = width * height * sizeof(unsigned char);

    // allocate device memory
    unsigned char* d_input;
    unsigned char* d_grayscale;
    unsigned char* d_edges;
    cudaError_t err;

    err = cudaMalloc((void**)&d_input, input_size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: Failed to allocate device memory for input image. " 
                  << cudaGetErrorString(err) << std::endl;
        stbi_image_free(h_input);
        return EXIT_FAILURE;
    }

    err = cudaMalloc((void**)&d_grayscale, grayscale_size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: Failed to allocate device memory for grayscale image. " 
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        stbi_image_free(h_input);
        return EXIT_FAILURE;
    }

    err = cudaMalloc((void**)&d_edges, edges_size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: Failed to allocate device memory for edges image. " 
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_grayscale);
        stbi_image_free(h_input);
        return EXIT_FAILURE;
    }

    // copy input image to device
    err = cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: Failed to copy input image to device. " 
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_grayscale);
        cudaFree(d_edges);
        stbi_image_free(h_input);
        return EXIT_FAILURE;
    }

    // Def CUDA kernel launch parameters for grayscale conversion
    int total_pixels = width * height;
    int threads_per_block = 256;
    int blocks_per_grid = (total_pixels + threads_per_block - 1) / threads_per_block;

    // launch CUDA kernel for grayscale conversion
    rgb_to_grayscale<<<blocks_per_grid, threads_per_block>>>(d_input, d_grayscale, width, height, channels);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: Grayscale kernel launch failed. " 
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_grayscale);
        cudaFree(d_edges);
        stbi_image_free(h_input);
        return EXIT_FAILURE;
    }

    // allocate host memory for grayscale image
    unsigned char* h_grayscale = new unsigned char[width * height];

    // copy grayscale image back to host
    err = cudaMemcpy(h_grayscale, d_grayscale, grayscale_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: Failed to copy grayscale image to host. " 
                  << cudaGetErrorString(err) << std::endl;
        delete[] h_grayscale;
        cudaFree(d_input);
        cudaFree(d_grayscale);
        cudaFree(d_edges);
        stbi_image_free(h_input);
        return EXIT_FAILURE;
    }

    if (!stbi_write_png(output_file_gray.c_str(), width, height, 1, h_grayscale, width)) {
        std::cerr << "Failed to write grayscale image: " << output_file_gray << std::endl;
    } else {
        std::cout << "Saved grayscale image as: " << output_file_gray << std::endl;
    }

    delete[] h_grayscale;

    // define CUDA kernel launch parameters for Sobel filter
    dim3 threadsPerBlock2(16, 16);
    dim3 blocksPerGrid2((width + threadsPerBlock2.x - 1) / threadsPerBlock2.x,
                        (height + threadsPerBlock2.y - 1) / threadsPerBlock2.y);

    // launch CUDA kernel for Sobel edge detection
    sobel_filter<<<blocksPerGrid2, threadsPerBlock2>>>(d_grayscale, d_edges, width, height);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: Sobel filter kernel launch failed. " 
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_grayscale);
        cudaFree(d_edges);
        stbi_image_free(h_input);
        return EXIT_FAILURE;
    }

    // allocate host memory for edges image
    unsigned char* h_edges = new unsigned char[width * height];

    // copy edges image back to host
    err = cudaMemcpy(h_edges, d_edges, edges_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: Failed to copy edges image to host. " 
                  << cudaGetErrorString(err) << std::endl;
        delete[] h_edges;
        cudaFree(d_input);
        cudaFree(d_grayscale);
        cudaFree(d_edges);
        stbi_image_free(h_input);
        return EXIT_FAILURE;
    }

    if (!stbi_write_png(output_file_edges.c_str(), width, height, 1, h_edges, width)) {
        std::cerr << "Failed to write edges image: " << output_file_edges << std::endl;
    } else {
        std::cout << "Saved edges image as: " << output_file_edges << std::endl;
    }

    delete[] h_edges;

    cudaFree(d_input);
    cudaFree(d_grayscale);
    cudaFree(d_edges);

    stbi_image_free(h_input);

    std::cout << "Image processing completed." << std::endl;
    return 0;
}
