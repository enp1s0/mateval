#include <cmath>
#include <cuda_fp16.h>
#include <mateval/cuda/norm.hpp>
#include <mateval/common.hpp>

namespace {
template <class T>
__global__ void sum_kernel (
	double* const sum_ptr,
	const T* const array_ptr,
	const std::size_t len
	) {
	const auto tid = blockDim.x * blockIdx.x + threadIdx.x;

	using acc_t = typename mtk::mateval::accumulate_t<T>::type;

	acc_t local_sum(0.);
	for (std::size_t i = tid; i < len; i += gridDim.x * blockDim.x) {
		acc_t a(array_ptr[i]);
		local_sum = local_sum + a * a;
	}

	atomicAdd(sum_ptr, static_cast<double>(local_sum));
}
} // unnamed namespace

template <class T>
double mtk::mateval::cuda::norm(
	const T* const array_ptr,
	const std::size_t len
	) {
	const auto block_size = 256;
	const auto grid_size = std::min(256lu, (len + block_size - 1) / block_size);

	double* d_sum_ptr;
	cudaMalloc(&d_sum_ptr, sizeof(double));
	cudaMemset(d_sum_ptr, 0, sizeof(double));

	cudaDeviceSynchronize();
	sum_kernel<<<grid_size, block_size>>>(
			d_sum_ptr,
			array_ptr,
			len
		);
	cudaDeviceSynchronize();

	double sum;
	cudaMemcpy(&sum, d_sum_ptr, sizeof(double), cudaMemcpyDefault);
	cudaFree(d_sum_ptr);

	return std::sqrt(sum);
}

template double mtk::mateval::cuda::norm<half  > (const half  * const, const std::size_t);
template double mtk::mateval::cuda::norm<float > (const float * const, const std::size_t);
template double mtk::mateval::cuda::norm<double> (const double* const, const std::size_t);
