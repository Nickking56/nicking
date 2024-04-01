#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <cmath>
#include <iostream>

const int DIMS[] = { 2, 5, 10, 25 };
const int NUM_DIMS = sizeof(DIMS) / sizeof(int);

// CUDA kernel for matrix multiplication adapted to different TILE_WIDTH
template <int TILE_WIDTH>
__global__ void tileMatrixMult(float* resultMat, const float* matA, const float* matB, int dimension) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < dimension && col < dimension) {
        float val = 0;
        for (int e = 0; e < dimension; ++e) {
            val += matA[row * dimension + e] * matB[e * dimension + col];
        }
        resultMat[row * dimension + col] = val;
    }
}

// Function to multiply matrices on the CPU for verification
void cpuMatrixMul(float* product, const float* matX, const float* matY, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float sum = 0;
            for (int k = 0; k < size; ++k) {
                sum += matX[i * size + k] * matY[k * size + j];
            }
            product[i * size + j] = sum;
        }
    }
}

// Function to compare the GPU and CPU results
bool checkResults(float* cpuRes, float* gpuRes, int size) {
    const float eps = 1e-5;
    for (int i = 0; i < size * size; i++) {
        if (fabs(cpuRes[i] - gpuRes[i]) > eps) {
            return false;
        }
    }
    return true;
}

int main() {
    int dim = 100; // Example dimension
    size_t bytes = dim * dim * sizeof(float);

    float* mat1 = (float*)malloc(bytes), * mat2 = (float*)malloc(bytes);
    float* cpuProd = (float*)malloc(bytes), * gpuProd = (float*)malloc(bytes);

    // Initialize matrices with random values
    srand(static_cast<unsigned>(time(nullptr)));
    for (int i = 0; i < dim * dim; ++i) {
        mat1[i] = static_cast<float>(rand()) / RAND_MAX;
        mat2[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    for (int i = 0; i < NUM_DIMS; ++i) {
        int tileSize = DIMS[i];
        float* dMat1, * dMat2, * dProd;
        cudaMalloc(&dMat1, bytes);
        cudaMalloc(&dMat2, bytes);
        cudaMalloc(&dProd, bytes);

        cudaMemcpy(dMat1, mat1, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(dMat2, mat2, bytes, cudaMemcpyHostToDevice);

        dim3 block(tileSize, tileSize);
        dim3 grid((dim + tileSize - 1) / tileSize, (dim + tileSize - 1) / tileSize);

        cudaEvent_t start, stop;
        float milliseconds = 0;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        // Direct kernel launch with template
        switch (tileSize) {
        case 2:
            tileMatrixMult<2> << <grid, block >> > (dProd, dMat1, dMat2, dim);
            break;
        case 5:
            tileMatrixMult<5> << <grid, block >> > (dProd, dMat1, dMat2, dim);
            break;
        case 10:
            tileMatrixMult<10> << <grid, block >> > (dProd, dMat1, dMat2, dim);
            break;
        case 25:
            tileMatrixMult<25> << <grid, block >> > (dProd, dMat1, dMat2, dim);
            break;
        default:
            printf("Unsupported TILE_WIDTH\n");
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);

        cudaMemcpy(gpuProd, dProd, bytes, cudaMemcpyDeviceToHost);

        cpuMatrixMul(cpuProd, mat1, mat2, dim);

        if (checkResults(cpuProd, gpuProd, dim)) {
            //std::cout << "TILE_WIDTH = " << tileSize << ": Test PASSED, GPU execution time: " << milliseconds << " ms" << std::endl;
        }
        else {
            //std::cout << "TILE_WIDTH = " << tileSize << ": Test FAILED" << std::endl;
        }
        cudaDeviceProp dp;
        cudaGetDeviceProperties(&dp, 0);
        printf("Threads That Can Be Simultaneously Scheduled = %d\n\n", dp.multiProcessorCount * dp.maxThreadsPerMultiProcessor);

        // Question 2: Assuming you've chosen a TILE_WIDTH and have a corresponding specialized kernel
        cudaFuncAttributes attr;
        // Here, you'd need a specialized version of your kernel to query attributes, e.g., tileMatrixMult<10>
        switch (tileSize) {
        case 2:
            cudaFuncGetAttributes(&attr, (void*)tileMatrixMult<2>); // This is how you'd theoretically call it for a specific specialization
            break;
        case 5:
            cudaFuncGetAttributes(&attr, (void*)tileMatrixMult<5>); // This is how you'd theoretically call it for a specific specialization
            break;
        case 10:
            cudaFuncGetAttributes(&attr, (void*)tileMatrixMult<10>); // This is how you'd theoretically call it for a specific specialization
            break;
        case 25:
            cudaFuncGetAttributes(&attr, (void*)tileMatrixMult<25>); // This is how you'd theoretically call it for a specific specialization
            break;
        default:
            printf("Unsupported TILE_WIDTH\n");
        }
        printf("Number of Registers = %d\n", attr.numRegs);
        printf("Shared Memory Size = %d bytes\n", 2 * tileSize * tileSize * 4);
        printf("Number of Blocks Per Streaming Multiprocessor = %d\n", dp.maxThreadsPerMultiProcessor / attr.maxThreadsPerBlock);
        printf("Maximum Total Threads Simultaneously Scheduled/Executing = %d\n", dp.maxThreadsPerMultiProcessor);

        cudaFree(dMat1);
        cudaFree(dMat2);
        cudaFree(dProd);
    }

    free(mat1);
    free(mat2);
    free(cpuProd);
    free(gpuProd);

    return 0;
}
