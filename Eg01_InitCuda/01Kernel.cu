#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define DATA_SIZE 1048576

int data[DATA_SIZE];

//initialization
bool InitCuda()
{
	int count{ 0 };

	cudaGetDeviceCount(&count);
	if (count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	int i{ 0 };
	for (int i = 0; i < count; ++i) {
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if (prop.major >= 1)
				break;
		}
	}

	if (i == count) {
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}

	cudaSetDevice(i);
	return true;

}

//generate numbers 0-9
void GenerateNumbers(int* number, int size)
{
	for (int i = 0; i < size; ++i)
	{
		number[i] = rand() % 10;
	}
}

//calculate sum of squares
__global__ static void sumOfSquares(int* num, int* result, clock_t* time)
{
	const int tid = threadIdx.x;

	int sum = 0;
	clock_t start = clock();

	for (int i = 0; i < DATA_SIZE; ++i)
	{
		sum += num[i] * num[i] * num[i];
	}
	
	*result = sum;
	*time = clock() - start;
}

int main()
{
	if (!InitCuda())
		return -1;
	
	//generate random numbers
	GenerateNumbers(data, DATA_SIZE);

	for (int n = 0; n < 20; ++n)
	{
		//copy data to GPU memory
		int* gpuData, *result;
		clock_t* time;
		cudaMalloc((void**)&gpuData, sizeof(int) * DATA_SIZE);
		cudaMalloc((void**)&result, sizeof(int));
		cudaMalloc((void**)&time, sizeof(clock_t));
		cudaMemcpy(gpuData, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice);

		//calculate in GPU
		sumOfSquares << <1, 1, 0 >> > (gpuData, result, time);
		int sum;
		clock_t time_use;
		cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&time_use, time, sizeof(clock_t), cudaMemcpyDeviceToHost);
		//free memory
		cudaFree(gpuData);
		cudaFree(result);
		cudaFree(time);
		printf("GPU Sum = %d, Time = %d\n", sum, time_use);
	}

	//calculate in CPU
	int sum = 0;
	for (int i = 0; i < DATA_SIZE; ++i)
		sum += data[i] * data[i] * data[i];
	printf("CPU Sum = %d\n", sum);
	
	system("pause");
	return 0;
}
