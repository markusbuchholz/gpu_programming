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
nvcc vector_add.cu -o vector_add

./vector_add

```

## Links

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [CUDA - samples](https://github.com/NVIDIA/cuda-samples)
- [CUDA lessons](https://github.com/ENCCS/cuda?tab=readme-ov-file)
- [cuda_programming](https://github.com/CoffeeBeforeArch/cuda_programming)
- [GPU-course](https://github.com/EPCCed/archer-gpu-course?tab=readme-ov-file)