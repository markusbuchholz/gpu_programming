# GPU Programming

## This repository contains sample programs that demonstrate the CUDA programming style.

### Check CUDA  Drivers and compiler

```bash
nvidia-smi

nvcc --vesion
```

### Install Nvidia Cuda compiler,

```bash
sudo apt install nvidia-cuda-toolkit
```

### Compile and run CUDA program

```bash

nvcc -std=c++20 -o vector_add vector_add.cu

./vector_add

```


### Architecture

SM (Streaming Multiprocessor) as one of the GPU’s “cores” (or compute units). Typicaly 56.

```bash
Host launch → Grid (many Blocks) 
                     ↓
             ┌───────────────┐
   SM₀ ─────▶│ Block A (256) │  
             │ Block B (256) │  ──▶ up to 2 048 active threads total (N = 256)
   SM₁ ─────▶│ Block C (256) │  
             │ Block D (256) │  
             └───────────────┘

N = 32, 64, 128, 256, 512, 1024.
```

1. Decide obout your number of elements ```N```, and number of threads per block ```NUM_THREADS```.

2. Compute ```NUM_BLOCKS```:

```bash
const int N            = 10000;
const int NUM_THREADS  = 512;
const int NUM_BLOCKS   = (N + NUM_THREADS - 1) / NUM_THREADS;  
//            = (10000 + 512 - 1) / 512
//            = 10511 / 512
//            = 20  (integer division)

```

3. Call function 

```bash

add<<<NUM_BLOCKS, NUM_THREADS>>>();
// add<<<20, 512>>>( /* your arguments, e.g. pointers and N */ );
```


## Links

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CUDA - samples](https://github.com/NVIDIA/cuda-samples)
- [CUDA lessons](https://github.com/ENCCS/cuda?tab=readme-ov-file)
- [cuda_programming](https://github.com/CoffeeBeforeArch/cuda_programming)
- [GPU-course](https://github.com/EPCCed/archer-gpu-course?tab=readme-ov-file)
- [GPU in robotics](https://github.com/JanuszBedkowski/gpu_computing_in_robotics)
- [introduction-cuda](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
- [tutorial-multi-gpu](https://github.com/FZJ-JSC/tutorial-multi-gpu?tab=readme-ov-file)
- [Cuda - Oxford](https://people.maths.ox.ac.uk/gilesm/cuda/index.html)
- [prof-Mike Giles-Oxford](https://people.maths.ox.ac.uk/gilesm/)