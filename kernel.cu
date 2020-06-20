#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>

#define N 50000           //在数组中定义元素的数量

__global__ void gpuAdd(int* d_a, int* d_b, int* d_c) {
	//给出当前内核的索引
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid<N)
	{
		d_c[tid] = d_a[tid] + d_b[tid];
		tid += blockDim.x * gridDim.x;
	}
}


int main(void) {
	//声明主机和设备的数组
	int h_a[N], h_b[N], h_c[N];
	int* d_a, * d_b, * d_c;

	//在设备上分配内存
	cudaMalloc((void**)&d_a, N * sizeof(int));
	cudaMalloc((void**)&d_b, N * sizeof(int));
	cudaMalloc((void**)&d_c, N * sizeof(int));

	//初始化主机数组
	for (int i = 0; i < N; i++)
	{
		h_a[i] = 2 * i * i;
		h_b[i] = i;
	}

	cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);
	
	//内核调用
	gpuAdd << <512, 512 >> > (d_a, d_b, d_c);
	cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);
	//下句代码确保内核执行在继续完成之前

	cudaDeviceSynchronize();
	int Correct = 1;
	printf("Vector addition an GPU\n");
	for (int i = 0; i < N; i++)
	{
		if ((h_a[i] + h_b[i] != h_c[i])) {
			Correct = 0;
		}
	}
	if (Correct == 1)
	{
		printf("GPU has computed Sum Correctly\n");
	}
	else
	{
		printf("There is an Error in GPU Compuation\n");
	}

	//释放内存
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return 0;
}