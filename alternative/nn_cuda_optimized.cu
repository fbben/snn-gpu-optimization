// Optimized parallel neural network using CUDA with advanced optimizations
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

// ! Network Parameters
#define INPUT_SIZE      32      // Number of input features
#define HIDDEN_SIZE     256     // Number of neurons in the hidden layer
#define OUTPUT_SIZE     1       // Number of output neurons
#define EPOCHS          100     // Number of training epochs
#define LOG_EVERY_EPOCH 1       // Log loss every n epochs
#define LEARNING_RATE   0.002
#define BATCH_SIZE      256     // Batch size for SGD
#define TILE_SIZE       16      // Tile size for shared memory optimization

// ! Data Structures
typedef struct {
    int rows;
    int cols;
    float *data;
} Matrix;

// GPU memory structure to hold persistent weights
typedef struct {
    float *d_W1;        // Device pointer for W1
    float *d_W2;        // Device pointer for W2
    float *d_X_batch;   // Device pointer for input batch
    float *d_Y_batch;   // Device pointer for target batch
    float *d_Z1;        // Device pointer for hidden layer output
    float *d_Y_pred;    // Device pointer for predictions
    float *d_dZ1;       // Device pointer for hidden layer gradients
    float *d_dZ2;       // Device pointer for output layer gradients
    float *d_dW1;       // Device pointer for W1 gradients
    float *d_dW2;       // Device pointer for W2 gradients
} GPUMemory;

// ! Memory Management
Matrix* allocate_matrix(int rows, int cols) {
    Matrix *m = (Matrix*)malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = (float*)malloc(rows * cols * sizeof(float));
    return m;
}

void free_matrix(Matrix *m) {
    free(m->data);
    free(m);
}

// ! Matrix Initialization
void random_init(Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->data[i * m->cols + j] = (float)rand() / RAND_MAX;
        }
    }
}

// ! OPTIMIZED CUDA KERNELS

// Kernel 1: Tiled Matrix Multiplication with Shared Memory
__global__ void mat_mult_tiled_kernel(float *A, float *B, float *C, 
                                      int A_rows, int A_cols, int B_cols) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float value = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (A_cols + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile from A
        if (row < A_rows && (t * TILE_SIZE + threadIdx.x) < A_cols)
            tile_A[threadIdx.y][threadIdx.x] = A[row * A_cols + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        
        // Load tile from B
        if ((t * TILE_SIZE + threadIdx.y) < A_cols && col < B_cols)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * B_cols + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++)
            value += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        
        __syncthreads();
    }
    
    if (row < A_rows && col < B_cols)
        C[row * B_cols + col] = value;
}

// Kernel 2: Fused ReLU Activation (in-place)
__global__ void relu_kernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// Kernel 3: Fused ReLU Derivative
__global__ void relu_derivative_kernel(float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (input[idx] > 0.0f) ? 1.0f : 0.0f;
    }
}

// Kernel 4: Compute dZ2 = (Y_pred - Y_batch) * (2.0 / batch_size)
__global__ void compute_dZ2_kernel(float *Y_pred, float *Y_batch, float *dZ2, 
                                   int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dZ2[idx] = (Y_pred[idx] - Y_batch[idx]) * scale;
    }
}

// Kernel 5: Element-wise multiplication for gradient masking
__global__ void elementwise_mult_kernel(float *A, float *B, float *C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * B[idx];
    }
}

// Kernel 6: Matrix Transpose
__global__ void transpose_kernel(float *input, float *output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

// Kernel 7: Weight Update
__global__ void update_weights_kernel(float *W, float *grad, float learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        W[idx] -= learning_rate * grad[idx];
    }
}

// Kernel 8: Compute MSE Loss
__global__ void compute_mse_kernel(float *Y_pred, float *Y_true, float *partial_loss, 
                                   int size) {
    __shared__ float shared_loss[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    float loss = 0.0f;
    if (idx < size) {
        float diff = Y_pred[idx] - Y_true[idx];
        loss = diff * diff;
    }
    shared_loss[tid] = loss;
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_loss[tid] += shared_loss[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_loss[blockIdx.x] = shared_loss[0];
    }
}

// ! GPU Memory Management Functions

GPUMemory* allocate_gpu_memory() {
    GPUMemory *gpu_mem = (GPUMemory*)malloc(sizeof(GPUMemory));
    
    // Allocate persistent weight matrices
    cudaMalloc((void**)&gpu_mem->d_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc((void**)&gpu_mem->d_W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    
    // Allocate batch buffers
    cudaMalloc((void**)&gpu_mem->d_X_batch, BATCH_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&gpu_mem->d_Y_batch, BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&gpu_mem->d_Z1, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc((void**)&gpu_mem->d_Y_pred, BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    
    // Allocate gradient buffers
    cudaMalloc((void**)&gpu_mem->d_dZ1, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc((void**)&gpu_mem->d_dZ2, BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&gpu_mem->d_dW1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc((void**)&gpu_mem->d_dW2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    
    return gpu_mem;
}

void free_gpu_memory(GPUMemory *gpu_mem) {
    cudaFree(gpu_mem->d_W1);
    cudaFree(gpu_mem->d_W2);
    cudaFree(gpu_mem->d_X_batch);
    cudaFree(gpu_mem->d_Y_batch);
    cudaFree(gpu_mem->d_Z1);
    cudaFree(gpu_mem->d_Y_pred);
    cudaFree(gpu_mem->d_dZ1);
    cudaFree(gpu_mem->d_dZ2);
    cudaFree(gpu_mem->d_dW1);
    cudaFree(gpu_mem->d_dW2);
    free(gpu_mem);
}

void initialize_weights_on_gpu(GPUMemory *gpu_mem, Matrix *W1, Matrix *W2) {
    cudaMemcpy(gpu_mem->d_W1, W1->data, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), 
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_mem->d_W2, W2->data, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), 
               cudaMemcpyHostToDevice);
}

// ! Optimized Forward Pass (entirely on GPU)
void forward_pass_gpu(GPUMemory *gpu_mem, int batch_size) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    
    // Z1 = X_batch * W1 (using tiled multiplication)
    dim3 numBlocks1((HIDDEN_SIZE + TILE_SIZE - 1) / TILE_SIZE,
                    (batch_size + TILE_SIZE - 1) / TILE_SIZE);
    mat_mult_tiled_kernel<<<numBlocks1, threadsPerBlock>>>(
        gpu_mem->d_X_batch, gpu_mem->d_W1, gpu_mem->d_Z1,
        batch_size, INPUT_SIZE, HIDDEN_SIZE);
    
    // Apply ReLU activation (fused kernel)
    int size_Z1 = batch_size * HIDDEN_SIZE;
    int blocks_relu = (size_Z1 + 255) / 256;
    relu_kernel<<<blocks_relu, 256>>>(gpu_mem->d_Z1, size_Z1);
    
    // Y_pred = Z1 * W2 (using tiled multiplication)
    dim3 numBlocks2((OUTPUT_SIZE + TILE_SIZE - 1) / TILE_SIZE,
                    (batch_size + TILE_SIZE - 1) / TILE_SIZE);
    mat_mult_tiled_kernel<<<numBlocks2, threadsPerBlock>>>(
        gpu_mem->d_Z1, gpu_mem->d_W2, gpu_mem->d_Y_pred,
        batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
}

// ! Optimized Backward Pass (entirely on GPU)
void backward_pass_gpu(GPUMemory *gpu_mem, int batch_size) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    
    // Compute dZ2 = (Y_pred - Y_batch) * (2.0 / batch_size)
    int size_dZ2 = batch_size * OUTPUT_SIZE;
    int blocks_dZ2 = (size_dZ2 + 255) / 256;
    compute_dZ2_kernel<<<blocks_dZ2, 256>>>(
        gpu_mem->d_Y_pred, gpu_mem->d_Y_batch, gpu_mem->d_dZ2,
        size_dZ2, 2.0f / batch_size);
    
    // Allocate temporary memory for transposed matrices
    float *d_Z1_T, *d_W2_T, *d_X_batch_T, *d_relu_deriv;
    cudaMalloc((void**)&d_Z1_T, HIDDEN_SIZE * batch_size * sizeof(float));
    cudaMalloc((void**)&d_W2_T, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc((void**)&d_X_batch_T, INPUT_SIZE * batch_size * sizeof(float));
    cudaMalloc((void**)&d_relu_deriv, batch_size * HIDDEN_SIZE * sizeof(float));
    
    // Transpose Z1
    dim3 trans_blocks1((batch_size + TILE_SIZE - 1) / TILE_SIZE,
                       (HIDDEN_SIZE + TILE_SIZE - 1) / TILE_SIZE);
    transpose_kernel<<<trans_blocks1, threadsPerBlock>>>(
        gpu_mem->d_Z1, d_Z1_T, batch_size, HIDDEN_SIZE);
    
    // Compute dW2 = Z1^T * dZ2
    dim3 dW2_blocks((OUTPUT_SIZE + TILE_SIZE - 1) / TILE_SIZE,
                    (HIDDEN_SIZE + TILE_SIZE - 1) / TILE_SIZE);
    mat_mult_tiled_kernel<<<dW2_blocks, threadsPerBlock>>>(
        d_Z1_T, gpu_mem->d_dZ2, gpu_mem->d_dW2,
        HIDDEN_SIZE, batch_size, OUTPUT_SIZE);
    
    // Update W2
    int size_W2 = HIDDEN_SIZE * OUTPUT_SIZE;
    int blocks_W2 = (size_W2 + 255) / 256;
    update_weights_kernel<<<blocks_W2, 256>>>(
        gpu_mem->d_W2, gpu_mem->d_dW2, LEARNING_RATE, size_W2);
    
    // Transpose W2
    dim3 trans_blocks2((HIDDEN_SIZE + TILE_SIZE - 1) / TILE_SIZE,
                       (OUTPUT_SIZE + TILE_SIZE - 1) / TILE_SIZE);
    transpose_kernel<<<trans_blocks2, threadsPerBlock>>>(
        gpu_mem->d_W2, d_W2_T, HIDDEN_SIZE, OUTPUT_SIZE);
    
    // Compute dZ1 = dZ2 * W2^T
    dim3 dZ1_blocks((HIDDEN_SIZE + TILE_SIZE - 1) / TILE_SIZE,
                    (batch_size + TILE_SIZE - 1) / TILE_SIZE);
    mat_mult_tiled_kernel<<<dZ1_blocks, threadsPerBlock>>>(
        gpu_mem->d_dZ2, d_W2_T, gpu_mem->d_dZ1,
        batch_size, OUTPUT_SIZE, HIDDEN_SIZE);
    
    // Compute ReLU derivative (before ReLU was applied, we need the pre-activation values)
    // For simplicity, we'll use the post-activation Z1 for the derivative
    int size_dZ1 = batch_size * HIDDEN_SIZE;
    int blocks_relu_deriv = (size_dZ1 + 255) / 256;
    relu_derivative_kernel<<<blocks_relu_deriv, 256>>>(
        gpu_mem->d_Z1, d_relu_deriv, size_dZ1);
    
    // Apply ReLU derivative: dZ1 = dZ1 * relu_derivative(Z1)
    elementwise_mult_kernel<<<blocks_relu_deriv, 256>>>(
        gpu_mem->d_dZ1, d_relu_deriv, gpu_mem->d_dZ1, size_dZ1);
    
    // Transpose X_batch
    dim3 trans_blocks3((batch_size + TILE_SIZE - 1) / TILE_SIZE,
                       (INPUT_SIZE + TILE_SIZE - 1) / TILE_SIZE);
    transpose_kernel<<<trans_blocks3, threadsPerBlock>>>(
        gpu_mem->d_X_batch, d_X_batch_T, batch_size, INPUT_SIZE);
    
    // Compute dW1 = X_batch^T * dZ1
    dim3 dW1_blocks((HIDDEN_SIZE + TILE_SIZE - 1) / TILE_SIZE,
                    (INPUT_SIZE + TILE_SIZE - 1) / TILE_SIZE);
    mat_mult_tiled_kernel<<<dW1_blocks, threadsPerBlock>>>(
        d_X_batch_T, gpu_mem->d_dZ1, gpu_mem->d_dW1,
        INPUT_SIZE, batch_size, HIDDEN_SIZE);
    
    // Update W1
    int size_W1 = INPUT_SIZE * HIDDEN_SIZE;
    int blocks_W1 = (size_W1 + 255) / 256;
    update_weights_kernel<<<blocks_W1, 256>>>(
        gpu_mem->d_W1, gpu_mem->d_dW1, LEARNING_RATE, size_W1);
    
    // Free temporary memory
    cudaFree(d_Z1_T);
    cudaFree(d_W2_T);
    cudaFree(d_X_batch_T);
    cudaFree(d_relu_deriv);
}

// ! Compute Loss on GPU
float compute_loss_gpu(GPUMemory *gpu_mem, int batch_size) {
    int size = batch_size * OUTPUT_SIZE;
    int num_blocks = (size + 255) / 256;
    
    float *d_partial_loss;
    cudaMalloc((void**)&d_partial_loss, num_blocks * sizeof(float));
    
    compute_mse_kernel<<<num_blocks, 256>>>(
        gpu_mem->d_Y_pred, gpu_mem->d_Y_batch, d_partial_loss, size);
    
    // Copy partial results back and sum on CPU
    float *h_partial_loss = (float*)malloc(num_blocks * sizeof(float));
    cudaMemcpy(h_partial_loss, d_partial_loss, num_blocks * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    float total_loss = 0.0f;
    for (int i = 0; i < num_blocks; i++) {
        total_loss += h_partial_loss[i];
    }
    
    free(h_partial_loss);
    cudaFree(d_partial_loss);
    
    return total_loss / batch_size;
}

// ! Batch Processing
void get_batch(Matrix *X, Matrix *Y, Matrix *X_batch, Matrix *Y_batch, 
               int batch_start, int batch_size) {
    for(int i = 0; i < batch_size; i++) {
        for(int j = 0; j < INPUT_SIZE; j++)
            X_batch->data[i * INPUT_SIZE + j] = X->data[(batch_start + i) * INPUT_SIZE + j];
        Y_batch->data[i * OUTPUT_SIZE] = Y->data[(batch_start + i) * OUTPUT_SIZE];
    }
}

// ! Data Loading
int load_csv(const char *filename, Matrix **X, Matrix **Y, int *num_samples) {
    FILE *file = fopen(filename, "r");
    if(!file) {
        printf("Failed to open file.\n");
        return -1;
    }
    char line[1024];
    int count = 0;
    while(fgets(line, sizeof(line), file)) count++;
    *num_samples = count;
    rewind(file);
    
    *X = allocate_matrix(count, INPUT_SIZE);
    *Y = allocate_matrix(count, OUTPUT_SIZE);
    
    int i = 0;
    while(fgets(line, sizeof(line), file)) {
        char *token = strtok(line, ",");
        int j = 0;
        while(token) {
            if(j < INPUT_SIZE) {
                (*X)->data[i * INPUT_SIZE + j] = atof(token);
            } else {
                (*Y)->data[i * OUTPUT_SIZE] = atof(token);
            }
            j++;
            token = strtok(NULL, ",");
        }
        i++;
    }
    fclose(file);
    return 0;
}

// ! Main Function
int main(int argc, char *argv[]) {
    if(argc != 2) {
        printf("Usage: %s <data.csv>\n", argv[0]);
        return -1;
    }

    double start_time, end_time;

    // Load data
    Matrix *X, *Y;
    int num_samples;
    if(load_csv(argv[1], &X, &Y, &num_samples) != 0)
        return -1;

    // Initialize weights on CPU
    Matrix *W1 = allocate_matrix(INPUT_SIZE, HIDDEN_SIZE);
    Matrix *W2 = allocate_matrix(HIDDEN_SIZE, OUTPUT_SIZE);
    random_init(W1);
    random_init(W2);

    // Allocate GPU memory (persistent)
    GPUMemory *gpu_mem = allocate_gpu_memory();
    
    // Copy weights to GPU (done once at the beginning)
    initialize_weights_on_gpu(gpu_mem, W1, W2);

    printf("Starting optimized training with GPU-persistent weights...\n");
    start_time = omp_get_wtime();

    // Training loop
    for(int epoch = 0; epoch < EPOCHS; epoch++) {
        for(int batch_start = 0; batch_start < num_samples; batch_start += BATCH_SIZE) {
            int batch_end = fmin(batch_start + BATCH_SIZE, num_samples);
            int batch_size = batch_end - batch_start;

            // Extract batch on CPU
            Matrix *X_batch = allocate_matrix(batch_size, INPUT_SIZE);
            Matrix *Y_batch = allocate_matrix(batch_size, OUTPUT_SIZE);
            get_batch(X, Y, X_batch, Y_batch, batch_start, batch_size);

            // Transfer batch to GPU
            cudaMemcpy(gpu_mem->d_X_batch, X_batch->data, 
                       batch_size * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_mem->d_Y_batch, Y_batch->data, 
                       batch_size * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

            // Forward pass (entirely on GPU)
            forward_pass_gpu(gpu_mem, batch_size);

            // Compute loss
            if((batch_start == 0) && ((epoch % LOG_EVERY_EPOCH == 0 && epoch != 0) || 
                epoch == 1 || epoch == EPOCHS - 1)) {
                float loss = compute_loss_gpu(gpu_mem, batch_size);
                printf("Epoch %d, MSE: %f\n", epoch, loss);
            }

            // Backward pass (entirely on GPU)
            backward_pass_gpu(gpu_mem, batch_size);

            // Synchronize to ensure all GPU operations are complete
            cudaDeviceSynchronize();

            // Free CPU batch memory
            free_matrix(X_batch);
            free_matrix(Y_batch);
        }
    }

    end_time = omp_get_wtime();
    printf("Training time: %.4f seconds\n", end_time - start_time);

    // Cleanup
    free_gpu_memory(gpu_mem);
    free_matrix(W1);
    free_matrix(W2);
    free_matrix(X);
    free_matrix(Y);

    return 0;
}
