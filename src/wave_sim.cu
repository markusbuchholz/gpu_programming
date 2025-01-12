// wave_simulation.cu

#include <GL/glew.h>          // Must be included before other OpenGL headers
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>  // CUDA-OpenGL interop
#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <string>
#include <filesystem> // C++17 feature
#include <cstring>
#include <cmath>

// Simulation parameters
const int WIDTH = 512;
const int HEIGHT = 512;
const float C = 0.1f;    // Wave speed
const float DT = 0.1f;   // Time step
const float DX = 1.0f;   // Spatial step

// CUDA error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// CUDA kernel to update wave state
__global__ void update_wave(float* u_prev, float* u_curr, float* u_next, int width, int height, float c, float dt, float dx)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height) return;

    int idx = y * width + x;

    // Neighbors indices with boundary checks
    float u_left   = (x > 0) ? u_curr[y * width + (x - 1)] : 0.0f;
    float u_right  = (x < width -1) ? u_curr[y * width + (x + 1)] : 0.0f;
    float u_up     = (y > 0) ? u_curr[(y -1) * width + x] : 0.0f;
    float u_down   = (y < height -1) ? u_curr[(y +1) * width + x] : 0.0f;

    // Finite difference approximation
    float laplacian = (u_left + u_right + u_up + u_down - 4.0f * u_curr[idx]) / (dx * dx);

    // Wave equation finite difference update
    u_next[idx] = 2.0f * u_curr[idx] - u_prev[idx] + (c * c) * (dt * dt) * laplacian;
}

// Initialize wave state with a disturbance in the center
void initialize_wave(std::vector<float>& u_prev, std::vector<float>& u_curr, int width, int height)
{
    // Set all to 0
    std::fill(u_prev.begin(), u_prev.end(), 0.0f);
    std::fill(u_curr.begin(), u_curr.end(), 0.0f);

    // Create a Gaussian disturbance in the center
    int center_x = width / 2;
    int center_y = height / 2;
    float sigma = 10.0f;

    for(int y = 0; y < height; ++y)
    {
        for(int x = 0; x < width; ++x)
        {
            int idx = y * width + x;
            float dx_val = x - center_x;
            float dy_val = y - center_y;
            u_prev[idx] = u_curr[idx] = expf(-(dx_val*dx_val + dy_val*dy_val) / (2.0f * sigma * sigma));
        }
    }
}

int main() {
    // Initialize GLFW
    if(!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    // Create window
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "CUDA Wave Simulation", NULL, NULL);
    if(!window)
    {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if(glewInit() != GLEW_OK)
    {
        std::cerr << "Failed to initialize GLEW\n";
        return -1;
    }

    // Create OpenGL texture
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    // Allocate texture storage (single channel for grayscale)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, WIDTH, HEIGHT, 0, GL_RED, GL_FLOAT, NULL);

    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Create Pixel Buffer Object (PBO) for CUDA-OpenGL interop
    GLuint pbo;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT * sizeof(float), NULL, GL_DYNAMIC_DRAW);

    // Register PBO with CUDA
    cudaGraphicsResource* cuda_pbo;
    cudaCheckError(cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsMapFlagsWriteDiscard));

    // Initialize wave state
    std::vector<float> h_u_prev(WIDTH * HEIGHT);
    std::vector<float> h_u_curr(WIDTH * HEIGHT);
    initialize_wave(h_u_prev, h_u_curr, WIDTH, HEIGHT);

    // Allocate device memory for wave simulation
    float *d_u_prev, *d_u_curr, *d_u_next;
    cudaCheckError(cudaMalloc((void**)&d_u_prev, WIDTH * HEIGHT * sizeof(float)));
    cudaCheckError(cudaMalloc((void**)&d_u_curr, WIDTH * HEIGHT * sizeof(float)));
    cudaCheckError(cudaMalloc((void**)&d_u_next, WIDTH * HEIGHT * sizeof(float)));

    // Copy initial state to device
    cudaCheckError(cudaMemcpy(d_u_prev, h_u_prev.data(), WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_u_curr, h_u_curr.data(), WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice));

    // Define CUDA kernel launch parameters for wave update
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid( (WIDTH + threads_per_block.x -1)/threads_per_block.x,
                          (HEIGHT + threads_per_block.y -1)/threads_per_block.y );

    // Main loop
    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // Update wave state
        update_wave<<<blocks_per_grid, threads_per_block>>>(d_u_prev, d_u_curr, d_u_next, WIDTH, HEIGHT, C, DT, DX);
        cudaCheckError(cudaGetLastError());

        // Swap buffers
        std::swap(d_u_prev, d_u_curr);
        std::swap(d_u_curr, d_u_next);

        // Map PBO for CUDA
        float* d_image;
        size_t num_bytes;
        cudaCheckError(cudaGraphicsMapResources(1, &cuda_pbo, 0));
        cudaCheckError(cudaGraphicsResourceGetMappedPointer((void**)&d_image, &num_bytes, cuda_pbo));

        // Copy current wave state to PBO
        cudaCheckError(cudaMemcpy(d_image, d_u_curr, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToDevice));

        // Unmap PBO
        cudaCheckError(cudaGraphicsUnmapResources(1, &cuda_pbo, 0));

        // Update texture with PBO data
        glBindTexture(GL_TEXTURE_2D, texture);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RED, GL_FLOAT, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // Render a quad with the texture
        glClear(GL_COLOR_BUFFER_BIT);

        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, texture);

        glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
            glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f, -1.0f);
            glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f,  1.0f);
            glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f,  1.0f);
        glEnd();

        glDisable(GL_TEXTURE_2D);

        glfwSwapBuffers(window);
    }

    // Cleanup
    cudaGraphicsUnregisterResource(cuda_pbo);
    glDeleteBuffers(1, &pbo);
    glDeleteTextures(1, &texture);

    cudaFree(d_u_prev);
    cudaFree(d_u_curr);
    cudaFree(d_u_next);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
