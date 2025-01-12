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