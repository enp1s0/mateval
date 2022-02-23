#include <mateval/comparison_cuda.hpp>
#include <cmath>
#include <cuda_fp16.h>

namespace {
constexpr unsigned block_size = 256;

template <class A_T, class B_T, class R_T>
__global__ void error_AxB_kernel(
		const mtk::mateval::error_t error_type,
		double* const result_ptr,
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
	double error = 0.0;
	double element = 0.0;

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
		error = abs(diff);
		element = abs(sum);
	}

	double* my_result_ptr = result_ptr;
	if (error_type & mtk::mateval::max_relative_error) {
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
		__syncthreads();
		if (threadIdx.x == 0) {
			my_result_ptr[blockIdx.x] = smem_diff[0];
			my_result_ptr += gridDim.x;
			my_result_ptr[blockIdx.x] = smem_base[0];
			my_result_ptr += gridDim.x;
		}
		__syncthreads();
	}
	if (error_type & (mtk::mateval::max_relative_error | mtk::mateval::max_absolute_error)) {
		__shared__ double smem_error[block_size];
		smem_error[threadIdx.x] = error;

		__syncthreads();
		for (unsigned i = block_size / 2; i >= 1; i >>= 1) {
			if (threadIdx.x < i) {
				const auto i0 = threadIdx.x;
				const auto i1 = i0 + i;
				smem_error[i0] = max(smem_error[i0], smem_error[i1]);
			}
			__syncthreads();
		}
		if (threadIdx.x == 0) {
			my_result_ptr[blockIdx.x] = smem_error[0];
			my_result_ptr += gridDim.x;
		}
	}
	if (error_type & mtk::mateval::max_relative_error) {
		__shared__ double smem_element[block_size];
		smem_element[threadIdx.x] = element;

		__syncthreads();
		for (unsigned i = block_size / 2; i >= 1; i >>= 1) {
			if (threadIdx.x < i) {
				const auto i0 = threadIdx.x;
				const auto i1 = i0 + i;
				smem_element[i0] = max(smem_element[i0], smem_element[i1]);
			}
			__syncthreads();
		}
		if (threadIdx.x == 0) {
			my_result_ptr[blockIdx.x] = smem_element[0];
			my_result_ptr += gridDim.x;
		}
	}
}
} // noname namespace

template <class A_T, class B_T, class REF_T>
std::unordered_map<mtk::mateval::error_t, double> mtk::mateval::cuda::get_error_AxB(
		const mtk::mateval::error_t error,
		const unsigned M, const unsigned N, const unsigned K,
		const mtk::mateval::major_t a_major, const mtk::mateval::major_t b_major, const mtk::mateval::major_t r_major,
		const A_T*   const a_ptr, const unsigned lda,
		const B_T*   const b_ptr, const unsigned ldb,
		const REF_T* const r_ptr, const unsigned ldr
		) {
	const auto num_threads = M * N;
	const auto grid_size = (num_threads + block_size - 1) / block_size;

	unsigned num_result_elements = 0;
	if (error & mtk::mateval::relative_residual) {
		num_result_elements += 2;
	}
	if (error & (mtk::mateval::max_absolute_error | mtk::mateval::max_relative_error)) {
		num_result_elements += 1;
	}
	if (error & mtk::mateval::max_relative_error) {
		num_result_elements += 1;
	}

	double *h_result;
	cudaMallocHost(&h_result, grid_size * sizeof(double) * num_result_elements);
#pragma omp paralell for
	for (unsigned i = 0; i < grid_size * num_result_elements; i++) {
		h_result[i] = 0.;
	}

	error_AxB_kernel<A_T, B_T, REF_T><<<grid_size, block_size>>>(
			error,
			h_result,
			M, N, K,
			a_major, b_major, r_major,
			a_ptr, lda,
			b_ptr, ldb,
			r_ptr, ldr
			);
	cudaDeviceSynchronize();

	std::unordered_map<mtk::mateval::error_t, double> result;

	double max_error = 0.0;
	double max_element = 0.0;
	double base_norm = 0.0;
	double diff_norm = 0.0;
	double *tmp_result_ptr = h_result;
	if (error & mtk::mateval::relative_residual) {
#pragma omp parallel for reduction(+: base_norm) reduction(+: diff_norm)
		for (unsigned i = 0; i < grid_size; i++) {
			diff_norm += tmp_result_ptr[i];
		}
		tmp_result_ptr += grid_size;
#pragma omp parallel for reduction(+: base_norm) reduction(+: diff_norm)
		for (unsigned i = 0; i < grid_size; i++) {
			base_norm += tmp_result_ptr[i];
		}
		tmp_result_ptr += grid_size;

		result.insert(std::make_pair(mtk::mateval::relative_residual, std::sqrt(diff_norm / base_norm)));
	}
	if (error & (mtk::mateval::max_absolute_error | mtk::mateval::max_relative_error)) {
#pragma omp parallel for reduction(max: max_error)
		for (unsigned i = 0; i < grid_size; i++) {
			max_error = std::max(max_error, tmp_result_ptr[i]);
		}
		tmp_result_ptr += grid_size;
	}
	if (error & mtk::mateval::max_absolute_error) {
		result.insert(std::make_pair(mtk::mateval::max_absolute_error, (max_error)));
	}
	if (error & mtk::mateval::max_relative_error) {
#pragma omp parallel for reduction(max: max_element)
		for (unsigned i = 0; i < grid_size; i++) {
			max_element = std::max(max_element, tmp_result_ptr[i]);
		}
		tmp_result_ptr += grid_size;
		result.insert(std::make_pair(mtk::mateval::max_relative_error, (max_error / max_element)));
	}

	cudaFreeHost(h_result);

	return result;
}

template std::unordered_map<mtk::mateval::error_t, double> mtk::mateval::cuda::get_error_AxB<half  , half  , half  >(const mtk::mateval::error_t, const unsigned, const unsigned, const unsigned, const mtk::mateval::major_t, const mtk::mateval::major_t, const mtk::mateval::major_t, const half  * const, const unsigned, const half  * const, const unsigned, const half  * const, const unsigned);
template std::unordered_map<mtk::mateval::error_t, double> mtk::mateval::cuda::get_error_AxB<float , float , float >(const mtk::mateval::error_t, const unsigned, const unsigned, const unsigned, const mtk::mateval::major_t, const mtk::mateval::major_t, const mtk::mateval::major_t, const float * const, const unsigned, const float * const, const unsigned, const float * const, const unsigned);
template std::unordered_map<mtk::mateval::error_t, double> mtk::mateval::cuda::get_error_AxB<double, double, double>(const mtk::mateval::error_t, const unsigned, const unsigned, const unsigned, const mtk::mateval::major_t, const mtk::mateval::major_t, const mtk::mateval::major_t, const double* const, const unsigned, const double* const, const unsigned, const double* const, const unsigned);

// SVD
template <class U_T, class S_T, class V_T, class REF_T>
__global__ void residual_SVD_kernel(
		double* const diff_norm,
		double* const base_norm,
		const unsigned M, const unsigned N, const unsigned K,
		const mtk::mateval::major_t u_major, const mtk::mateval::major_t v_major, const mtk::mateval::major_t r_major,
		const U_T*   const u_ptr, const unsigned ldu,
		const S_T*   const s_ptr,
		const V_T*   const v_ptr, const unsigned ldv,
		const REF_T* const r_ptr, const unsigned ldr
		) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

	double sum = 0.0;
	double diff = 0.0;
	double base = 0.0;

	if (tid < M * N) {
		const auto row = tid % M;
		const auto col = tid / M;

		for (unsigned k = 0; k < K; k++) {
			std::size_t u_index;
			if (u_major == mtk::mateval::col_major) {
				u_index = row + k * ldu;
			} else {
				u_index = k + row * ldu;
			}

			std::size_t v_index;
			if (v_major == mtk::mateval::col_major) {
				v_index = col + k * ldv;
			} else {
				v_index = k + col * ldv;
			}

			const auto du = static_cast<double>(u_ptr[u_index]);
			const auto dv = static_cast<double>(v_ptr[v_index]);
			const auto ds = static_cast<double>(s_ptr[k]);
			sum = fma(du * ds, dv, sum);
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

template <class U_T, class S_T, class V_T, class REF_T>
double mtk::mateval::cuda::residual_UxSxVt(
		const unsigned M, const unsigned N, const unsigned K,
		const mtk::mateval::major_t u_major, const mtk::mateval::major_t v_major, const mtk::mateval::major_t r_major,
		const U_T*   const u_ptr, const unsigned ldu,
		const S_T*   const s_ptr,
		const V_T*   const v_ptr, const unsigned ldv,
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

	residual_SVD_kernel<U_T, S_T, V_T, REF_T><<<grid_size, block_size>>>(
			h_diff, h_base,
			M, N, K,
			u_major, v_major, r_major,
			u_ptr, ldu,
			s_ptr,
			v_ptr, ldv,
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

template double mtk::mateval::cuda::residual_UxSxVt<double, double, double, double>(const unsigned, const unsigned, const unsigned, const mtk::mateval::major_t, const mtk::mateval::major_t, const mtk::mateval::major_t, const double* const, const unsigned, const double* const, const double* const, const unsigned, const double* const, const unsigned);
template double mtk::mateval::cuda::residual_UxSxVt<float , float , float , float >(const unsigned, const unsigned, const unsigned, const mtk::mateval::major_t, const mtk::mateval::major_t, const mtk::mateval::major_t, const float * const, const unsigned, const float * const, const float * const, const unsigned, const float * const, const unsigned);
template double mtk::mateval::cuda::residual_UxSxVt<half  , half  , half  , half  >(const unsigned, const unsigned, const unsigned, const mtk::mateval::major_t, const mtk::mateval::major_t, const mtk::mateval::major_t, const half  * const, const unsigned, const half  * const, const half  * const, const unsigned, const half  * const, const unsigned);

template <class T>
__global__ void orthogonality_kernel(
		double* const diff_norm,
		const unsigned M, const unsigned N,
		const mtk::mateval::major_t major,
		const T* const ptr, const unsigned ld
		) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

	double diff = 0;
	if (tid < N * N) {
		const auto row = tid % N;
		const auto col = tid / N;

		double sum = 0.;
		for (unsigned k = 0; k < M; k++) {
			std::size_t a_index;
			std::size_t b_index;
			if (major == mtk::mateval::col_major) {
				a_index = k + row * ld;
				b_index = k + col * ld;
			} else {
				a_index = row + k * ld;
				b_index = col + k * ld;
			}

			const auto da = static_cast<double>(ptr[a_index]);
			const auto db = static_cast<double>(ptr[b_index]);
			sum = fma(da, db, sum);
		}

		double base = (row == col) ? 1.0 : 0.0;
		diff = sum - base;
	}

	__shared__ double smem_diff[block_size];
	smem_diff[threadIdx.x] = diff * diff;

	__syncthreads();
	for (unsigned i = block_size / 2; i >= 1; i >>= 1) {
		if (threadIdx.x < i) {
			const auto i0 = threadIdx.x;
			const auto i1 = i0 + i;
			smem_diff[i0] += smem_diff[i1];
		}
		__syncthreads();
	}
	diff_norm[blockIdx.x] = smem_diff[0];
}

template <class T>
double mtk::mateval::cuda::orthogonality(
		const unsigned M, const unsigned N,
		const mtk::mateval::major_t major,
		const T* const ptr, const unsigned ld
		) {
	const auto num_threads = N * N;
	const auto grid_size = (num_threads + block_size - 1) / block_size;

	double *h_diff;
	cudaMallocHost(&h_diff, grid_size * sizeof(double));
	for (unsigned i = 0; i < grid_size; i++) {
		h_diff[i] = 0.;
	}

	orthogonality_kernel<T><<<grid_size, block_size>>>(
			h_diff,
			M, N,
			major,
			ptr, ld
			);
	cudaDeviceSynchronize();

	double diff_norm = 0.0;
#pragma omp parallel for reduction(+: diff_norm)
	for (unsigned i = 0; i < grid_size; i++) {
		diff_norm += h_diff[i];
	}

	cudaFreeHost(h_diff);

	return std::sqrt(diff_norm / N);
}

template double mtk::mateval::cuda::orthogonality<double>(const unsigned, const unsigned, const mtk::mateval::major_t, const double* const, const unsigned);
template double mtk::mateval::cuda::orthogonality<float >(const unsigned, const unsigned, const mtk::mateval::major_t, const float * const, const unsigned);
template double mtk::mateval::cuda::orthogonality<half  >(const unsigned, const unsigned, const mtk::mateval::major_t, const half  * const, const unsigned);
