#include <cfloat>
#include <cuda_fp16.h>
#include <mateval/cuda/minmax.hpp>

namespace {
template <class T>
__global__ void operate_kernel (
	double* const max_ptr,
	double* const min_ptr,
	double* const abs_max_ptr,
	double* const abs_min_ptr,
	const mtk::mateval::operation_t op,
	const mtk::mateval::layout_t layout,
	const std::size_t m,
	const std::size_t n,
	const T* const mat_ptr,
	const std::size_t ld
	) {
	const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= m * n) {
		return;
	}

	double local_max = DBL_MIN;
	double local_min = DBL_MAX;
	double local_abs_max = DBL_MIN;
	double local_abs_min = DBL_MAX;
	for (auto tid = blockDim.x * blockIdx.x + threadIdx.x; tid < m * n; tid += blockDim.x * gridDim.x) {
		const auto im = tid % m;
		const auto in = tid % n;

		std::size_t mem_index;
		if (layout == mtk::mateval::col_major) {
			mem_index = im + in * ld;
		} else {
			mem_index = in + im * ld;
		}

		const double v = mat_ptr[mem_index];

		if (op & mtk::mateval::op_abs_max)
			local_abs_max = max(local_abs_max, std::abs(v));
		if (op & mtk::mateval::op_abs_min)
			local_abs_min = min(local_abs_min, std::abs(v));
		if (op & mtk::mateval::op_max)
			local_max = max(local_max, v);
		if (op & mtk::mateval::op_min)
			local_min = min(local_min, v);
	}


	if (op & mtk::mateval::op_abs_max)
		atomicMax(reinterpret_cast<long long int*>(abs_max_ptr), *reinterpret_cast<long long int*>(&local_abs_max));
	if (op & mtk::mateval::op_abs_min)
		atomicMin(reinterpret_cast<long long int*>(abs_min_ptr), *reinterpret_cast<long long int*>(&local_abs_min));
	if (op & mtk::mateval::op_max)
		atomicMax(reinterpret_cast<long long int*>(max_ptr), *reinterpret_cast<long long int*>(&local_max));
	if (op & mtk::mateval::op_min)
		atomicMin(reinterpret_cast<long long int*>(min_ptr), *reinterpret_cast<long long int*>(&local_min));
}
} // unnamed namespace

template <class T>
mtk::mateval::error_map_t mtk::mateval::cuda::operate (
	const mtk::mateval::operation_t op,
	const mtk::mateval::layout_t layout,
	const std::size_t m,
	const std::size_t n,
	const T* const mat_ptr,
	const std::size_t ld
	) {

	double* max_ptr = nullptr;
	double* min_ptr = nullptr;
	double* abs_max_ptr = nullptr;
	double* abs_min_ptr = nullptr;

	constexpr auto grid_size = 256;
	constexpr auto block_size = 256;

	constexpr auto num_threads = grid_size * block_size;

	if (op & mtk::mateval::op_abs_max) {
		cudaMallocManaged(&abs_max_ptr, sizeof(double) * num_threads); *abs_max_ptr = DBL_MIN;
	}
	if (op & mtk::mateval::op_abs_min) {
		cudaMallocManaged(&abs_min_ptr, sizeof(double) * num_threads); *abs_min_ptr = DBL_MAX;
	}
	if (op & mtk::mateval::op_max) {
		cudaMallocManaged(&max_ptr, sizeof(double) * num_threads); *max_ptr = DBL_MIN;
	}
	if (op & mtk::mateval::op_min) {
		cudaMallocManaged(&min_ptr, sizeof(double) * num_threads); *min_ptr = DBL_MAX;
	}

	cudaDeviceSynchronize();
	operate_kernel<<<grid_size, block_size>>>(
		max_ptr, min_ptr,
		abs_max_ptr, abs_min_ptr,
		op,
		layout,
		m, n,
		mat_ptr,
		ld
		);
	cudaDeviceSynchronize();

	mtk::mateval::error_map_t res;
	if (op & mtk::mateval::op_abs_max) {
		res.insert(std::make_pair(mtk::mateval::op_abs_max, *abs_max_ptr));
		cudaFree(abs_max_ptr);
	}
	if (op & mtk::mateval::op_abs_min) {
		res.insert(std::make_pair(mtk::mateval::op_abs_min, *abs_min_ptr));
		cudaFree(abs_min_ptr);
	}
	if (op & mtk::mateval::op_max) {
		res.insert(std::make_pair(mtk::mateval::op_max, *max_ptr));
		cudaFree(max_ptr);
	}
	if (op & mtk::mateval::op_min) {
		res.insert(std::make_pair(mtk::mateval::op_min, *min_ptr));
		cudaFree(min_ptr);
	}

	return res;
}

template mtk::mateval::error_map_t mtk::mateval::cuda::operate<half  > (const mtk::mateval::operation_t, const mtk::mateval::layout_t, const std::size_t, const std::size_t, const half  * const, const std::size_t);
template mtk::mateval::error_map_t mtk::mateval::cuda::operate<float > (const mtk::mateval::operation_t, const mtk::mateval::layout_t, const std::size_t, const std::size_t, const float * const, const std::size_t);
template mtk::mateval::error_map_t mtk::mateval::cuda::operate<double> (const mtk::mateval::operation_t, const mtk::mateval::layout_t, const std::size_t, const std::size_t, const double* const, const std::size_t);
