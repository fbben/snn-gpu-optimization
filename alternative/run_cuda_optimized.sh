nvcc -o nn_cuda_optimized nn_cuda_optimized.cu -Xcompiler -fopenmp

# ./nn_cuda_optimized ../refrence/data/synthetic_convex_small.csv
# ./nn_cuda_optimized ../refrence/data/synthetic_convex_medium.csv
./nn_cuda_optimized ../refrence/data/synthetic_convex_large.csv
