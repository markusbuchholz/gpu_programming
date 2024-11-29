#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem> 
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace fs = std::filesystem;

__global__ void rgb_to_grayscale(unsigned char* d_input, unsigned char* d_output, int width, int height, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    if (idx < total_pixels) {
        int input_idx = idx * channels;
        unsigned char r = d_input[input_idx];
        unsigned char g = d_input[input_idx + 1];
        unsigned char b = d_input[input_idx + 2];

        unsigned char gray = static_cast<unsigned char>(0.21f * r + 0.72f * g + 0.07f * b);

        d_output[idx] = gray;
    }
}

int main() {
    std::string input_file = "../data/moon.png";   
    std::string output_file = "../data/moon_gray.png"; 

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
    size_t output_size = width * height * sizeof(unsigned char);

    // allocate device memory
    unsigned char* d_input;
    unsigned char* d_output;
    cudaError_t err;

    err = cudaMalloc((void**)&d_input, input_size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: Failed to allocate device memory for input image. " 
                  << cudaGetErrorString(err) << std::endl;
        stbi_image_free(h_input);
        return EXIT_FAILURE;
    }

    err = cudaMalloc((void**)&d_output, output_size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: Failed to allocate device memory for output image. " 
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        stbi_image_free(h_input);
        return EXIT_FAILURE;
    }

    // copy input image to device
    err = cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: Failed to copy input image to device. " 
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        stbi_image_free(h_input);
        return EXIT_FAILURE;
    }

    // define CUDA kernel launch parameters
    int total_pixels = width * height;
    int threads_per_block = 256;
    int blocks_per_grid = (total_pixels + threads_per_block - 1) / threads_per_block;

    // launch CUDA kernel
    rgb_to_grayscale<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, width, height, channels);

    // check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: Kernel launch failed. " 
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        stbi_image_free(h_input);
        return EXIT_FAILURE;
    }

    // allocate host memory for output
    unsigned char* h_output = new unsigned char[width * height];

    // copy output image back to host
    err = cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: Failed to copy output image to host. " 
                  << cudaGetErrorString(err) << std::endl;
        delete[] h_output;
        cudaFree(d_input);
        cudaFree(d_output);
        stbi_image_free(h_input);
        return EXIT_FAILURE;
    }

    if (!stbi_write_png(output_file.c_str(), width, height, 1, h_output, width)) {
        std::cerr << "Failed to write image: " << output_file << std::endl;
    } else {
        std::cout << "Saved grayscale image as: " << output_file << std::endl;
    }

    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    stbi_image_free(h_input);

    std::cout << "Image processing completed." << std::endl;
    return 0;
}
