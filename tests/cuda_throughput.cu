#include <iostream>
#include <chrono>
#include <mateval/cuda/comparison.hpp>

constexpr auto N = 1lu << 14;

void measure_throughput(
		const std::size_t m,
		const std::size_t n,
		const std::size_t k,
		const float* const a_ptr,
		const float* const b_ptr,
		const float* const c_ptr
		) {
	mtk::mateval::cuda::get_error_AxB(
			mtk::mateval::max_absolute_error,
			m, n, k,
			mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major,
			a_ptr, k,
			b_ptr, k,
			c_ptr, m
			);
	cudaDeviceSynchronize();
	const auto start_clock = std::chrono::system_clock::now();
	mtk::mateval::cuda::get_error_AxB(
			mtk::mateval::max_absolute_error,
			m, n, k,
			mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major,
			a_ptr, k,
			b_ptr, k,
			c_ptr, m
			);
	cudaDeviceSynchronize();
	const auto end_clock = std::chrono::system_clock::now();
	const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1e-6;
	const auto computational_complexity = 2 * m * n * k;
	printf("[m=%7lu, n=%7lu, k=%7lu] : r=%e [GFlop/s]\n",
			m, n, k,
			computational_complexity / elapsed_time * 1e-9
			);
}

__global__ void init_array_kernel(
		float* ptr,
		const std::size_t length
		) {
	const auto tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= length) return;

	ptr[tid] = tid / static_cast<float>(length);
}

void init_array(
		float* ptr,
		const std::size_t length
		) {
	constexpr std::size_t block_size = 256;
	const std::size_t grid_size = (length + block_size - 1) / block_size;

	init_array_kernel<<<grid_size, block_size>>>(
			ptr,
			length
			);
	cudaDeviceSynchronize();
}

int main() {
	float *da_ptr;
	float *db_ptr;
	float *dc_ptr;
	cudaMalloc(&da_ptr, sizeof(float) * N * N);
	cudaMalloc(&db_ptr, sizeof(float) * N * N);
	cudaMalloc(&dc_ptr, sizeof(float) * N * N);

	init_array(da_ptr, N * N);
	init_array(db_ptr, N * N);
	init_array(dc_ptr, N * N);

	measure_throughput(
			N, N, N,
			da_ptr, db_ptr, dc_ptr
			);

	cudaFree(da_ptr);
	cudaFree(db_ptr);
	cudaFree(dc_ptr);
}
