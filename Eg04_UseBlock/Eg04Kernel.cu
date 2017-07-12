#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define DATA_SIZE 1048576
#define THREAD_NUM 256
#define BLOCK_NUM 32

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
	const int bid = blockIdx.x;

	int sum = 0;
	if (tid == 0)
		time[bid] = clock();

	for (int i = bid * THREAD_NUM + tid; i < DATA_SIZE; i += BLOCK_NUM * THREAD_NUM)
	{
		sum += num[i] * num[i] * num[i];
	}

	result[bid * THREAD_NUM + tid] = sum;
	if (tid == 0)
		time[bid + BLOCK_NUM] = clock();
}

int main()
{
	if (!InitCuda())
		return -1;

	//generate random numbers
	GenerateNumbers(data, DATA_SIZE);

	clock_t	start = clock();
	for (int n = 0; n < 20; ++n)
	{
		//copy data to GPU memory
		int* gpuData, *result;
		clock_t* time;
		cudaMalloc((void**)&gpuData, sizeof(int) * DATA_SIZE);
		cudaMalloc((void**)&result, sizeof(int) * THREAD_NUM * BLOCK_NUM);
		cudaMalloc((void**)&time, sizeof(clock_t) * BLOCK_NUM * 2);
		cudaMemcpy(gpuData, data, sizeof(int) * DATA_SIZE, cudaMemcpyHostToDevice);

		//calculate in GPU
		sumOfSquares << <BLOCK_NUM, THREAD_NUM, 0 >> > (gpuData, result, time);

		int sum[THREAD_NUM * BLOCK_NUM];
		clock_t time_use[BLOCK_NUM * 2];
		cudaMemcpy(&sum, result, sizeof(int) * THREAD_NUM * BLOCK_NUM, cudaMemcpyDeviceToHost);
		cudaMemcpy(&time_use, time, sizeof(clock_t) * BLOCK_NUM * 2, cudaMemcpyDeviceToHost);
		//free memory
		cudaFree(gpuData);
		cudaFree(result);
		cudaFree(time);

		//calculate the sum
		int finnal_sum = 0;
		for (int i = 0; i < THREAD_NUM * BLOCK_NUM; ++i)
			finnal_sum += sum[i];

		//calculate the run time
		clock_t min_start, max_end;
		min_start = time_use[0];
		max_end = time_use[BLOCK_NUM];
		for (int i = 1; i < BLOCK_NUM; ++i)
		{
			if (min_start > time_use[i])
				min_start = time_use[i];
			if (max_end < time_use[i + BLOCK_NUM])
				max_end = time_use[i + BLOCK_NUM];
		}
		printf("GPU Sum = %d, Time = %d\n", finnal_sum, max_end - min_start);
	}
	printf("GPU Time = %d\n\n", clock() - start);

	//calculate in CPU
	int finnal_sum = 0;
	for (int i = 0; i < DATA_SIZE; ++i)
		finnal_sum += data[i] * data[i] * data[i];
	printf("CPU Sum = %d\n", finnal_sum);

	system("pause");
	return 0;
}
