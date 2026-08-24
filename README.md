# GPU Memory Optimization for Shallow Neural Network Training

A neural-network training project implemented in C, pthreads, CUDA, and optimized CUDA.

## Requirements

- GCC with OpenMP support
- NVIDIA CUDA Toolkit (`nvcc`) and a CUDA-capable GPU for CUDA versions

## Project layout

- `refrence/` — sequential, pthreads, and baseline CUDA implementations
- `alternative/` — optimized CUDA implementation
- `refrence/data/` — CSV datasets
- `project_paper.pdf` — project report

## Run

Run the commands from the indicated directory.

```bash
# Sequential C version
cd refrence
bash scripts/run.sh

# Pthreads version
bash scripts/run_pthreads.sh

# CUDA version
bash scripts/run_cuda.sh

# Optimized CUDA version
cd ../alternative
bash run_cuda_optimized.sh
```

Each script builds its program and trains it using `synthetic_convex_large.csv`.
