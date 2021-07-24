#include <mateval/comparison_cuda.hpp>
#include <cmath>
#include <cuda_fp16.h>

namespace {
constexpr unsigned block_size = 256;

template <class A_T, class B_T, class R_T>
__global__ void max_error_kernel(
		double* const max_error,
		const unsigned M, const unsigned N, const unsigned K,
		const mtk::mateval::major_t a_major, const mtk::mateval::major_t b_major, const mtk::mateval::major_t r_major,
		const A_T* const a_ptr, const unsigned lda,
		const B_T* const b_ptr, const unsigned ldb,
		const R_T* const r_ptr, const unsigned ldr
		) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

	double sum = 0.0;
	double diff = 0.0;

	if (tid < M * N) {
		const auto row = tid % M;
		const auto col = tid / M;

		for (unsigned k = 0; k < K; k++) {
			std::size_t a_index;
			if (a_major == mtk::mateval::col_major) {
				a_index = row + k * lda;
			} else {
				a_index = k + row * lda;
			}

			std::size_t b_index;
			if (b_major == mtk::mateval::col_major) {
				b_index = k + col * ldb;
			} else {
				b_index = col + k * ldb;
			}

			const auto da = static_cast<double>(a_ptr[a_index]);
			const auto db = static_cast<double>(b_ptr[b_index]);
			sum = fma(da, db, sum);
		}

		std::size_t r_index;
		if (r_major == mtk::mateval::col_major) {
			r_index = row + col * ldr;
		} else {
			r_index = col + row * ldr;
		}

		diff = abs(sum - static_cast<double>(r_ptr[r_index]));
	}

	__shared__ double smem[block_size];
	smem[threadIdx.x] = diff;

	__syncthreads();
	for (unsigned i = block_size / 2; i >= 1; i >>= 1) {
		if (threadIdx.x < i) {
			const auto i0 = threadIdx.x;
			const auto i1 = i0 + i;
			smem[i0] = max(smem[i0], smem[i1]);
		}
		__syncthreads();
	}
	max_error[blockIdx.x] = smem[0];
}

template <class A_T, class B_T, class R_T>
__global__ void max_relative_error_kernel(
		double* const max_relative_error,
		const unsigned M, const unsigned N, const unsigned K,
		const mtk::mateval::major_t a_major, const mtk::mateval::major_t b_major, const mtk::mateval::major_t r_major,
		const A_T* const a_ptr, const unsigned lda,
		const B_T* const b_ptr, const unsigned ldb,
		const R_T* const r_ptr, const unsigned ldr
		) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

	double sum = 0.0;
	double sum_abs = 0.0;
	double relative_error = 0.0;

	if (tid < M * N) {
		const auto row = tid % M;
		const auto col = tid / M;

		for (unsigned k = 0; k < K; k++) {
			std::size_t a_index;
			if (a_major == mtk::mateval::col_major) {
				a_index = row + k * lda;
			} else {
				a_index = k + row * lda;
			}

			std::size_t b_index;
			if (b_major == mtk::mateval::col_major) {
				b_index = k + col * ldb;
			} else {
				b_index = col + k * ldb;
			}

			const auto da = static_cast<double>(a_ptr[a_index]);
			const auto db = static_cast<double>(b_ptr[b_index]);
			sum = fma(da, db, sum);
			sum_abs = fma(abs(da), abs(db), sum_abs);
		}

		std::size_t r_index;
		if (r_major == mtk::mateval::col_major) {
			r_index = row + col * ldr;
		} else {
			r_index = col + row * ldr;
		}

		relative_error = abs((sum - static_cast<double>(r_ptr[r_index])) / sum_abs);
	}

	__shared__ double smem[block_size];
	smem[threadIdx.x] = relative_error;

	__syncthreads();
	for (unsigned i = block_size / 2; i >= 1; i >>= 1) {
		if (threadIdx.x < i) {
			const auto i0 = threadIdx.x;
			const auto i1 = i0 + i;
			smem[i0] = max(smem[i0], smem[i1]);
		}
		__syncthreads();
	}
	max_relative_error[blockIdx.x] = smem[0];
}

template <class A_T, class B_T, class R_T>
__global__ void residual_kernel(
		double* const diff_norm,
		double* const base_norm,
		const unsigned M, const unsigned N, const unsigned K,
		const mtk::mateval::major_t a_major, const mtk::mateval::major_t b_major, const mtk::mateval::major_t r_major,
		const A_T* const a_ptr, const unsigned lda,
		const B_T* const b_ptr, const unsigned ldb,
		const R_T* const r_ptr, const unsigned ldr
		) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

	double sum = 0.0;
	double diff = 0.0;
	double base = 0.0;

	if (tid < M * N) {
		const auto row = tid % M;
		const auto col = tid / M;

		for (unsigned k = 0; k < K; k++) {
			std::size_t a_index;
			if (a_major == mtk::mateval::col_major) {
				a_index = row + k * lda;
			} else {
				a_index = k + row * lda;
			}

			std::size_t b_index;
			if (b_major == mtk::mateval::col_major) {
				b_index = k + col * ldb;
			} else {
				b_index = col + k * ldb;
			}

			const auto da = static_cast<double>(a_ptr[a_index]);
			const auto db = static_cast<double>(b_ptr[b_index]);
			sum = fma(da, db, sum);
		}

		std::size_t r_index;
		if (r_major == mtk::mateval::col_major) {
			r_index = row + col * ldr;
		} else {
			r_index = col + row * ldr;
		}

		base = r_ptr[r_index];
		diff = sum - base;
	}

	__shared__ double smem_diff[block_size];
	__shared__ double smem_base[block_size];
	smem_diff[threadIdx.x] = diff * diff;
	smem_base[threadIdx.x] = base * base;

	__syncthreads();
	for (unsigned i = block_size / 2; i >= 1; i >>= 1) {
		if (threadIdx.x < i) {
			const auto i0 = threadIdx.x;
			const auto i1 = i0 + i;
			smem_base[i0] += smem_base[i1];
			smem_diff[i0] += smem_diff[i1];
		}
		__syncthreads();
	}
	diff_norm[blockIdx.x] = smem_diff[0];
	base_norm[blockIdx.x] = smem_base[0];
}

template <class A_T, class B_T, class R_T>
__global__ void max_error_and_residual_kernel(
		double* const max_error,
		double* const diff_norm,
		double* const base_norm,
		const unsigned M, const unsigned N, const unsigned K,
		const mtk::mateval::major_t a_major, const mtk::mateval::major_t b_major, const mtk::mateval::major_t r_major,
		const A_T* const a_ptr, const unsigned lda,
		const B_T* const b_ptr, const unsigned ldb,
		const R_T* const r_ptr, const unsigned ldr
		) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

	double sum = 0.0;
	double diff = 0.0;
	double base = 0.0;
	double diff_abs = 0.0;

	if (tid < M * N) {
		const auto row = tid % M;
		const auto col = tid / M;

		for (unsigned k = 0; k < K; k++) {
			std::size_t a_index;
			if (a_major == mtk::mateval::col_major) {
				a_index = row + k * lda;
			} else {
				a_index = k + row * lda;
			}

			std::size_t b_index;
			if (b_major == mtk::mateval::col_major) {
				b_index = k + col * ldb;
			} else {
				b_index = col + k * ldb;
			}

			const auto da = static_cast<double>(a_ptr[a_index]);
			const auto db = static_cast<double>(b_ptr[b_index]);
			sum = fma(da, db, sum);
		}

		std::size_t r_index;
		if (r_major == mtk::mateval::col_major) {
			r_index = row + col * ldr;
		} else {
			r_index = col + row * ldr;
		}

		base = r_ptr[r_index];
		diff = sum - base;
		diff_abs = abs(diff);
	}

	__shared__ double smem_diff[block_size];
	__shared__ double smem_base[block_size];
	__shared__ double smem_diff_abs[block_size];
	smem_diff[threadIdx.x] = diff * diff;
	smem_base[threadIdx.x] = base * base;
	smem_diff_abs[threadIdx.x] = diff_abs;

	__syncthreads();
	for (unsigned i = block_size / 2; i >= 1; i >>= 1) {
		if (threadIdx.x < i) {
			const auto i0 = threadIdx.x;
			const auto i1 = i0 + i;
			smem_base[i0] += smem_base[i1];
			smem_diff[i0] += smem_diff[i1];
			smem_diff_abs[i0] = max(smem_diff_abs[i0], smem_diff_abs[i1]);
		}
		__syncthreads();
	}
	diff_norm[blockIdx.x] = smem_diff[0];
	base_norm[blockIdx.x] = smem_base[0];
	max_error[blockIdx.x] = smem_diff_abs[0];
}

template <class A_T, class B_T, class R_T>
__global__ void max_relative_error_and_residual_kernel(
		double* const max_relative_error,
		double* const diff_norm,
		double* const base_norm,
		const unsigned M, const unsigned N, const unsigned K,
		const mtk::mateval::major_t a_major, const mtk::mateval::major_t b_major, const mtk::mateval::major_t r_major,
		const A_T* const a_ptr, const unsigned lda,
		const B_T* const b_ptr, const unsigned ldb,
		const R_T* const r_ptr, const unsigned ldr
		) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

	double sum = 0.0;
	double sum_abs = 0.0;
	double diff = 0.0;
	double base = 0.0;
	double relative_error = 0.0;

	if (tid < M * N) {
		const auto row = tid % M;
		const auto col = tid / M;

		for (unsigned k = 0; k < K; k++) {
			std::size_t a_index;
			if (a_major == mtk::mateval::col_major) {
				a_index = row + k * lda;
			} else {
				a_index = k + row * lda;
			}

			std::size_t b_index;
			if (b_major == mtk::mateval::col_major) {
				b_index = k + col * ldb;
			} else {
				b_index = col + k * ldb;
			}

			const auto da = static_cast<double>(a_ptr[a_index]);
			const auto db = static_cast<double>(b_ptr[b_index]);
			sum = fma(da, db, sum);
			sum_abs = fma(abs(da), abs(db), sum_abs);
		}

		std::size_t r_index;
		if (r_major == mtk::mateval::col_major) {
			r_index = row + col * ldr;
		} else {
			r_index = col + row * ldr;
		}

		base = r_ptr[r_index];
		diff = sum - base;
		relative_error = abs(diff / sum_abs);
	}

	__shared__ double smem_diff[block_size];
	__shared__ double smem_base[block_size];
	__shared__ double smem_relative_error[block_size];
	smem_diff[threadIdx.x] = diff * diff;
	smem_base[threadIdx.x] = base * base;
	smem_relative_error[threadIdx.x] = relative_error;

	__syncthreads();
	for (unsigned i = block_size / 2; i >= 1; i >>= 1) {
		if (threadIdx.x < i) {
			const auto i0 = threadIdx.x;
			const auto i1 = i0 + i;
			smem_base[i0] += smem_base[i1];
			smem_diff[i0] += smem_diff[i1];
			smem_relative_error[i0] = max(smem_relative_error[i0], smem_relative_error[i1]);
		}
		__syncthreads();
	}
	diff_norm[blockIdx.x] = smem_diff[0];
	base_norm[blockIdx.x] = smem_base[0];
	max_relative_error[blockIdx.x] = smem_relative_error[0];
}
} // noname namespace

template <class A_T, class B_T, class REF_T>
double mtk::mateval::cuda::residual_AxB(
		const unsigned M, const unsigned N, const unsigned K,
		const mtk::mateval::major_t a_major, const mtk::mateval::major_t b_major, const mtk::mateval::major_t r_major,
		const A_T*   const a_ptr, const unsigned lda,
		const B_T*   const b_ptr, const unsigned ldb,
		const REF_T* const r_ptr, const unsigned ldr
		) {
	const auto num_threads = M * N;
	const auto grid_size = (num_threads + block_size - 1) / block_size;

	double *h_base;
	double *h_diff;
	cudaMallocHost(&h_base, grid_size * sizeof(double));
	cudaMallocHost(&h_diff, grid_size * sizeof(double));
	for (unsigned i = 0; i < grid_size; i++) {
		h_diff[i] = 0.;
		h_base[i] = 0.;
	}

	residual_kernel<A_T, B_T, REF_T><<<grid_size, block_size>>>(
			h_diff, h_base,
			M, N, K,
			a_major, b_major, r_major,
			a_ptr, lda,
			b_ptr, ldb,
			r_ptr, ldr
			);
	cudaDeviceSynchronize();

	double base_norm = 0.0;
	double diff_norm = 0.0;
#pragma omp parallel for reduction(+: base_norm) reduction(+: diff_norm)
	for (unsigned i = 0; i < grid_size; i++) {
		base_norm += h_base[i];
		diff_norm += h_diff[i];
	}

	cudaFreeHost(h_base);
	cudaFreeHost(h_diff);

	return std::sqrt(diff_norm / base_norm);
}

template double mtk::mateval::cuda::residual_AxB<half  , half  , half  >(const unsigned, const unsigned, const unsigned, const mtk::mateval::major_t, const mtk::mateval::major_t, const mtk::mateval::major_t, const half  * const, const unsigned, const half  * const, const unsigned, const half  * const, const unsigned);
template double mtk::mateval::cuda::residual_AxB<float , float , float >(const unsigned, const unsigned, const unsigned, const mtk::mateval::major_t, const mtk::mateval::major_t, const mtk::mateval::major_t, const float * const, const unsigned, const float * const, const unsigned, const float * const, const unsigned);
template double mtk::mateval::cuda::residual_AxB<double, double, double>(const unsigned, const unsigned, const unsigned, const mtk::mateval::major_t, const mtk::mateval::major_t, const mtk::mateval::major_t, const double* const, const unsigned, const double* const, const unsigned, const double* const, const unsigned);

template <class A_T, class B_T, class REF_T>
double mtk::mateval::cuda::max_error_AxB(
		const unsigned M, const unsigned N, const unsigned K,
		const mtk::mateval::major_t a_major, const mtk::mateval::major_t b_major, const mtk::mateval::major_t r_major,
		const A_T*   const a_ptr, const unsigned lda,
		const B_T*   const b_ptr, const unsigned ldb,
		const REF_T* const r_ptr, const unsigned ldr
		) {
	const auto num_threads = M * N;
	const auto grid_size = (num_threads + block_size - 1) / block_size;

	double *h_max_errors;
	cudaMallocHost(&h_max_errors, grid_size * sizeof(double));
	for (unsigned i = 0; i < grid_size; i++) {
		h_max_errors[i] = 0.;
	}

	max_error_kernel<A_T, B_T, REF_T><<<grid_size, block_size>>>(
			h_max_errors,
			M, N, K,
			a_major, b_major, r_major,
			a_ptr, lda,
			b_ptr, ldb,
			r_ptr, ldr
			);
	cudaDeviceSynchronize();

	double max_error = 0.0;
#pragma omp parallel for reduction(max: max_error)
	for (unsigned i = 0; i < grid_size; i++) {
		max_error = std::max(max_error, h_max_errors[i]);
	}

	cudaFreeHost(h_max_errors);

	return max_error;
}

template double mtk::mateval::cuda::max_error_AxB<half  , half  , half  >(const unsigned, const unsigned, const unsigned, const mtk::mateval::major_t, const mtk::mateval::major_t, const mtk::mateval::major_t, const half  * const, const unsigned, const half  * const, const unsigned, const half  * const, const unsigned);
template double mtk::mateval::cuda::max_error_AxB<float , float , float >(const unsigned, const unsigned, const unsigned, const mtk::mateval::major_t, const mtk::mateval::major_t, const mtk::mateval::major_t, const float * const, const unsigned, const float * const, const unsigned, const float * const, const unsigned);
template double mtk::mateval::cuda::max_error_AxB<double, double, double>(const unsigned, const unsigned, const unsigned, const mtk::mateval::major_t, const mtk::mateval::major_t, const mtk::mateval::major_t, const double* const, const unsigned, const double* const, const unsigned, const double* const, const unsigned);

template <class A_T, class B_T, class REF_T>
double mtk::mateval::cuda::max_relative_error_AxB(
		const unsigned M, const unsigned N, const unsigned K,
		const mtk::mateval::major_t a_major, const mtk::mateval::major_t b_major, const mtk::mateval::major_t r_major,
		const A_T*   const a_ptr, const unsigned lda,
		const B_T*   const b_ptr, const unsigned ldb,
		const REF_T* const r_ptr, const unsigned ldr
		) {
	const auto num_threads = M * N;
	const auto grid_size = (num_threads + block_size - 1) / block_size;

	double *h_max_relative_errors;
	cudaMallocHost(&h_max_relative_errors, grid_size * sizeof(double));
	for (unsigned i = 0; i < grid_size; i++) {
		h_max_relative_errors[i] = 0.;
	}

	max_error_kernel<A_T, B_T, REF_T><<<grid_size, block_size>>>(
			h_max_relative_errors,
			M, N, K,
			a_major, b_major, r_major,
			a_ptr, lda,
			b_ptr, ldb,
			r_ptr, ldr
			);
	cudaDeviceSynchronize();

	double max_relative_error = 0.0;
#pragma omp parallel for reduction(max: max_error)
	for (unsigned i = 0; i < grid_size; i++) {
		max_relative_error = std::max(max_relative_error, h_max_relative_errors[i]);
	}

	cudaFreeHost(h_max_relative_errors);

	return max_relative_error;
}

template double mtk::mateval::cuda::max_relative_error_AxB<half  , half  , half  >(const unsigned, const unsigned, const unsigned, const mtk::mateval::major_t, const mtk::mateval::major_t, const mtk::mateval::major_t, const half  * const, const unsigned, const half  * const, const unsigned, const half  * const, const unsigned);
template double mtk::mateval::cuda::max_relative_error_AxB<float , float , float >(const unsigned, const unsigned, const unsigned, const mtk::mateval::major_t, const mtk::mateval::major_t, const mtk::mateval::major_t, const float * const, const unsigned, const float * const, const unsigned, const float * const, const unsigned);
template double mtk::mateval::cuda::max_relative_error_AxB<double, double, double>(const unsigned, const unsigned, const unsigned, const mtk::mateval::major_t, const mtk::mateval::major_t, const mtk::mateval::major_t, const double* const, const unsigned, const double* const, const unsigned, const double* const, const unsigned);

template <class A_T, class B_T, class REF_T>
std::tuple<double, double> mtk::mateval::cuda::max_error_and_residual_AxB(
		const unsigned M, const unsigned N, const unsigned K,
		const major_t a_major, const major_t b_major, const major_t r_major,
		const A_T*   const a_ptr, const unsigned lda,
		const B_T*   const b_ptr, const unsigned ldb,
		const REF_T* const r_ptr, const unsigned ldr
		) {
	const auto num_threads = M * N;
	const auto grid_size = (num_threads + block_size - 1) / block_size;

	double *h_base;
	double *h_diff;
	double *h_max_error;
	cudaMallocHost(&h_base, grid_size * sizeof(double));
	cudaMallocHost(&h_diff, grid_size * sizeof(double));
	cudaMallocHost(&h_max_error, grid_size * sizeof(double));
	for (unsigned i = 0; i < grid_size; i++) {
		h_diff[i] = 0.;
		h_base[i] = 0.;
		h_max_error[i] = 0.;
	}

	max_error_and_residual_kernel<A_T, B_T, REF_T><<<grid_size, block_size>>>(
			h_max_error,
			h_diff, h_base,
			M, N, K,
			a_major, b_major, r_major,
			a_ptr, lda,
			b_ptr, ldb,
			r_ptr, ldr
			);
	cudaDeviceSynchronize();

	double max_error = 0.0;
	double base_norm = 0.0;
	double diff_norm = 0.0;
#pragma omp parallel for reduction(+: base_norm) reduction(+: diff_norm) reduction(max: max_error)
	for (unsigned i = 0; i < grid_size; i++) {
		base_norm += h_base[i];
		diff_norm += h_diff[i];
		max_error = std::max(max_error, h_max_error[i]);
	}

	cudaFreeHost(h_base);
	cudaFreeHost(h_diff);
	cudaFreeHost(h_max_error);

	return std::tuple<double, double>(max_error, std::sqrt(diff_norm / base_norm));
}

template std::tuple<double, double> mtk::mateval::cuda::max_error_and_residual_AxB<half  , half  , half  >(const unsigned, const unsigned, const unsigned, const mtk::mateval::major_t, const mtk::mateval::major_t, const mtk::mateval::major_t, const half  * const, const unsigned, const half  * const, const unsigned, const half  * const, const unsigned);
template std::tuple<double, double> mtk::mateval::cuda::max_error_and_residual_AxB<float , float , float >(const unsigned, const unsigned, const unsigned, const mtk::mateval::major_t, const mtk::mateval::major_t, const mtk::mateval::major_t, const float * const, const unsigned, const float * const, const unsigned, const float * const, const unsigned);
template std::tuple<double, double> mtk::mateval::cuda::max_error_and_residual_AxB<double, double, double>(const unsigned, const unsigned, const unsigned, const mtk::mateval::major_t, const mtk::mateval::major_t, const mtk::mateval::major_t, const double* const, const unsigned, const double* const, const unsigned, const double* const, const unsigned);

template <class A_T, class B_T, class REF_T>
std::tuple<double, double> mtk::mateval::cuda::max_relative_error_and_residual_AxB(
		const unsigned M, const unsigned N, const unsigned K,
		const major_t a_major, const major_t b_major, const major_t r_major,
		const A_T*   const a_ptr, const unsigned lda,
		const B_T*   const b_ptr, const unsigned ldb,
		const REF_T* const r_ptr, const unsigned ldr
		) {
	const auto num_threads = M * N;
	const auto grid_size = (num_threads + block_size - 1) / block_size;

	double *h_base;
	double *h_diff;
	double *h_max_relative_error;
	cudaMallocHost(&h_base, grid_size * sizeof(double));
	cudaMallocHost(&h_diff, grid_size * sizeof(double));
	cudaMallocHost(&h_max_relative_error, grid_size * sizeof(double));
	for (unsigned i = 0; i < grid_size; i++) {
		h_diff[i] = 0.;
		h_base[i] = 0.;
		h_max_relative_error[i] = 0.;
	}

	max_relative_error_and_residual_kernel<A_T, B_T, REF_T><<<grid_size, block_size>>>(
			h_max_relative_error,
			h_diff, h_base,
			M, N, K,
			a_major, b_major, r_major,
			a_ptr, lda,
			b_ptr, ldb,
			r_ptr, ldr
			);
	cudaDeviceSynchronize();

	double max_relative_error = 0.0;
	double base_norm = 0.0;
	double diff_norm = 0.0;
#pragma omp parallel for reduction(+: base_norm) reduction(+: diff_norm) reduction(max: max_error)
	for (unsigned i = 0; i < grid_size; i++) {
		base_norm += h_base[i];
		diff_norm += h_diff[i];
		max_relative_error = std::max(max_relative_error, h_max_relative_error[i]);
	}

	cudaFreeHost(h_base);
	cudaFreeHost(h_diff);
	cudaFreeHost(h_max_relative_error);

	return std::tuple<double, double>(max_relative_error, std::sqrt(diff_norm / base_norm));
}

template std::tuple<double, double> mtk::mateval::cuda::max_relative_error_and_residual_AxB<half  , half  , half  >(const unsigned, const unsigned, const unsigned, const mtk::mateval::major_t, const mtk::mateval::major_t, const mtk::mateval::major_t, const half  * const, const unsigned, const half  * const, const unsigned, const half  * const, const unsigned);
template std::tuple<double, double> mtk::mateval::cuda::max_relative_error_and_residual_AxB<float , float , float >(const unsigned, const unsigned, const unsigned, const mtk::mateval::major_t, const mtk::mateval::major_t, const mtk::mateval::major_t, const float * const, const unsigned, const float * const, const unsigned, const float * const, const unsigned);
template std::tuple<double, double> mtk::mateval::cuda::max_relative_error_and_residual_AxB<double, double, double>(const unsigned, const unsigned, const unsigned, const mtk::mateval::major_t, const mtk::mateval::major_t, const mtk::mateval::major_t, const double* const, const unsigned, const double* const, const unsigned, const double* const, const unsigned);
