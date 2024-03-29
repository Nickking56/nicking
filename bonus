#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <iostream>
#include <cstdlib>
#include <ctime>

const int BLOCK_DIMS[] = { 2, 5, 10, 25 };
const int DIM_COUNT = sizeof(BLOCK_DIMS) / sizeof(int);

__global__ void multiplyMatrices(float* matPrimary, float* matSecondary, float* matResult, int primaryRows, int primaryCols, int secondaryCols) {
    int yIdx = blockIdx.y * blockDim.y + threadIdx.y;
    int xIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (yIdx < primaryRows && xIdx < secondaryCols) {
        float sum = 0;
        for (int index = 0; index < primaryCols; ++index) {
            sum += matPrimary[yIdx * primaryCols + index] * matSecondary[index * secondaryCols + xIdx];
        }
        matResult[yIdx * secondaryCols + xIdx] = sum;
    }
}

void cpuMatrixMultiplication(float* matrixA, float* matrixB, float* matrixC, int rowsMatrixA, int colsMatrixA, int colsMatrixB) {
    clock_t start = clock();

    for (int row = 0; row < rowsMatrixA; ++row) {
        for (int col = 0; col < colsMatrixB; ++col) {
            float accum = 0.0f;
            for (int k = 0; k < colsMatrixA; ++k) {
                accum += matrixA[row * colsMatrixA + k] * matrixB[k * colsMatrixB + col];
            }
            matrixC[row * colsMatrixB + col] = accum;
        }
    }

    clock_t end = clock();
    double cpuTimeUsed = ((double)(end - start)) / CLOCKS_PER_SEC;
}

void populateMatrix(float* mat, int height, int width) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            mat[i * width + j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }
}

bool matricesAreEqual(float* matrixOne, float* matrixTwo, int rows, int cols, float epsilon) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (fabs(matrixOne[i * cols + j] - matrixTwo[i * cols + j]) > epsilon) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    int testCases[2][3] = {
        {400, 450, 500},
        {1200, 1350, 1150}
    };

    for (int test = 0; test < 2; ++test) {
        int rowsFirstMatrix = testCases[test][0];
        int colsFirstMatrix = testCases[test][1];
        int colsSecondMatrix = testCases[test][2];

        size_t sizeFirst = rowsFirstMatrix * colsFirstMatrix * sizeof(float);
        size_t sizeSecond = colsFirstMatrix * colsSecondMatrix * sizeof(float);
        size_t sizeResult = rowsFirstMatrix * colsSecondMatrix * sizeof(float);

        float* firstMatrix, * secondMatrix, * resultMatrix;
        float* deviceFirst, * deviceSecond, * deviceResult;

        firstMatrix = (float*)malloc(sizeFirst);
        secondMatrix = (float*)malloc(sizeSecond);
        resultMatrix = (float*)malloc(sizeResult);

        populateMatrix(firstMatrix, rowsFirstMatrix, colsFirstMatrix);
        populateMatrix(secondMatrix, colsFirstMatrix, colsSecondMatrix);

        cudaMalloc(&deviceFirst, sizeFirst);
        cudaMalloc(&deviceSecond, sizeSecond);
        cudaMalloc(&deviceResult, sizeResult);

        cudaMemcpy(deviceFirst, firstMatrix, sizeFirst, cudaMemcpyHostToDevice);
        cudaMemcpy(deviceSecond, secondMatrix, sizeSecond, cudaMemcpyHostToDevice);

        for (int i = 0; i < DIM_COUNT; ++i) {
            int tileSize = BLOCK_DIMS[i];
            dim3 threadsPerBlock(tileSize, tileSize);
            dim3 blocksPerGrid((colsSecondMatrix + tileSize - 1) / tileSize, (rowsFirstMatrix + tileSize - 1) / tileSize);

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            multiplyMatrices << <blocksPerGrid, threadsPerBlock >> > (deviceFirst, deviceSecond, deviceResult, rowsFirstMatrix, colsFirstMatrix, colsSecondMatrix);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            cudaMemcpy(resultMatrix, deviceResult, sizeResult, cudaMemcpyDeviceToHost);

            float* verificationMatrix = (float*)malloc(sizeResult);
            cpuMatrixMultiplication(firstMatrix, secondMatrix, verificationMatrix, rowsFirstMatrix, colsFirstMatrix, colsSecondMatrix);

            if (matricesAreEqual(verificationMatrix, resultMatrix, rowsFirstMatrix, colsSecondMatrix, 1e-5f)) {
                std::cout << "Test Case " << test + 1 << " with TILE_SIZE " << tileSize << ": PASS: Execution Time " << milliseconds << " ms" << std::endl;
            }
            else {
                std::cout << "Test Case " << test + 1 << " with TILE_SIZE " << tileSize << ": FAIL" << std::endl;
            }

            free(verificationMatrix);
        }

        cudaFree(deviceFirst);
        cudaFree(deviceSecond);
        cudaFree(deviceResult);
        free(firstMatrix);
        free(secondMatrix);
        free(resultMatrix);
    }

    return 0;
}
