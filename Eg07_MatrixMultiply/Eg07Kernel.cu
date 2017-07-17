#include <stdio.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

#define THREAD_NUM 256
#define MAXTRIX_SIZE 1000
int block_num = MAXTRIX_SIZE * (MAXTRIX_SIZE + THREAD_NUM - 1) / THREAD_NUM;

//generate random matrix
void genMat(float* a, int n)
{
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
			a[i * n + j] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX * RAND_MAX);
	}
}

__global__ void matMult(const float* a, const float* b, float* c, int n)
{
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	const int idx = bid * THREAD_NUM + tid;

	const int row = idx / n;
	const int column = idx % n;

	if (row < n && column < n)
	{
		float t = 0;
		float y = 0;
		for (int i = 0; i < n; ++i)
		{
			float r;
			y -= a[row * n + i] * b[i * n + column];
			r = t - y;
			y = r - t + y;
			t = r;
			//t += a[row * n + i] * b[i * n + column];
		}
		c[row * n + column] = t;
	}
}


int main()
{
	float* a, *b, *c, *d;
	const int n = MAXTRIX_SIZE;

	//allocate memory
	a = (float*)malloc(sizeof(float) * n * n);
	b = (float*)malloc(sizeof(float) * n * n);
	c = (float*)malloc(sizeof(float) * n * n);
	d = (float*)malloc(sizeof(float) * n * n);	

	//generate random matrix
	genMat(a, n);
	genMat(b, n);

	//calculate by CPU
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			double t = 0;
			for (int k = 0; k < n; ++k)
				t += a[i * n + k] * b[k * n + j];
			d[i * n + j] = t;
		}
	}

	//copy to CUDA
	float* cuda_a, *cuda_b, *cuda_c;
	cudaMalloc((void**)&cuda_a, sizeof(float) * n * n);
	cudaMalloc((void**)&cuda_b, sizeof(float) * n * n);
	cudaMalloc((void**)&cuda_c, sizeof(float) * n * n);
	cudaMemcpy(cuda_a, a, sizeof(float) * n * n, cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_b, b, sizeof(float) * n * n, cudaMemcpyHostToDevice);

	//calculation
	matMult << <block_num, THREAD_NUM, 0 >> > (cuda_a, cuda_b, cuda_c, n);
	cudaMemcpy(c, cuda_c, sizeof(float) * n * n, cudaMemcpyDeviceToHost);

	//free memory
	cudaFree(cuda_a);
	cudaFree(cuda_b);
	cudaFree(cuda_c);
	
	//calculate the error
	double max_err = 0, avg_err = 0;
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			int idx = i * n + j;
			if (d[idx] != 0)
			{
				double err = fabs((c[idx] - d[idx]) / d[idx]);
				if (max_err < err)
					max_err = err;
				avg_err += err;
			}
		}
	}
	printf("Max Error = %g, Avg Error = %g\n", max_err, avg_err);



	system("pause");
	return 0;
}
