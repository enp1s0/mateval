#include <mateval/cuda/comparison.hpp>
#include <stdexcept>
#include <cmath>
#include <cuda_fp16.h>
#include <cuComplex.h>

namespace {
constexpr unsigned block_size = 256;

template <class T>
struct cmplx {
	using real_t = T;
	T x, y;

	__device__ __host__ cmplx(const T a) {x = a, y = 0;}
	__device__ __host__ cmplx(const T a, const T b) {x = a, y = b;}
};
template <class T>
__device__ __host__ cmplx<T> operator+(const cmplx<T> a, const cmplx<T> b){return cmplx<T>{a.x + b.x, a.y + b.y};}
template <class T>
__device__ __host__ cmplx<T> operator-(const cmplx<T> a, const cmplx<T> b){return cmplx<T>{a.x - b.x, a.y - b.y};}
template <class T>
__device__ __host__ cmplx<T> operator*(const cmplx<T> a, const cmplx<T> b){return cmplx<T>{a.x * b.x - a.y * b.y, a.y * b.x + b.y * a.x};}

__device__ __host__ cuComplex conj(const cuComplex& a){return cuComplex{a.x, -a.y};}
__device__ __host__ cuDoubleComplex conj(const cuDoubleComplex& a){return cuDoubleComplex{a.x, -a.y};}
__device__ __host__ half conj(const half& a){return a;}
__device__ __host__ float conj(const float& a){return a;}
__device__ __host__ double conj(const double& a){return a;}

template <class T>
struct rc_acc_type {using type = typename mtk::mateval::accumulate_t<T>::type;};
template <>
struct rc_acc_type<cuComplex      > {using type = cmplx<double>;};
template <>
struct rc_acc_type<cuDoubleComplex> {using type = cmplx<typename mtk::mateval::accumulate_t<double>::type>;};

template <class ACC_T, class T>
__device__ __host__ ACC_T cast2acc(const T a) {return static_cast<double>(a);}
template <class ACC_T>
__device__ __host__ ACC_T cast2acc(const cuComplex a) {return ACC_T{static_cast<typename ACC_T::real_t>(a.x), static_cast<typename ACC_T::real_t>(a.y)};}
template <class ACC_T>
__device__ __host__ ACC_T cast2acc(const cuDoubleComplex a) {return ACC_T{static_cast<typename ACC_T::real_t>(a.x), static_cast<typename ACC_T::real_t>(a.y)};}

template <class T>
__device__ __host__ double norm2(const T a) {return a * a;};
template <class T>
__device__ __host__ double norm2(const cmplx<T> a) {return a.x * a.x + a.y * a.y;};

template <class T>
__device__ __host__ double absmax(const T a) {return abs(static_cast<double>(a));};
template <class T>
__device__ __host__ double absmax(const cmplx<T> a) {return max(abs(static_cast<double>(a.x)), abs(static_cast<double>(a.y)));};

template <class T>
__device__ __host__ double relative_error(const T diff, const T base) {return (static_cast<double>(base) != 0.) ? abs(static_cast<double>(diff) / static_cast<double>(base)) : 0;};
template <class T>
__device__ __host__ double relative_error(const cmplx<T> diff, const cmplx<T> base) {return norm2(base) != 0 ? sqrt(norm2(diff) / norm2(base)) : 0;};

template <class A_T, class B_T, class C_T, class R_T, class ALPHA_T, class BETA_T>
__global__ void error_GEMM_kernel(
		const mtk::mateval::error_t error_type,
		double* const result_ptr,
		const unsigned M, const unsigned N, const unsigned K,
		const mtk::mateval::layout_t a_major, const mtk::mateval::layout_t b_major, const mtk::mateval::layout_t c_major, const mtk::mateval::layout_t r_major,
		const ALPHA_T alpha,
		const A_T* const a_ptr, const unsigned lda,
		const B_T* const b_ptr, const unsigned ldb,
		const BETA_T beta,
		const C_T* const c_ptr, const unsigned ldc,
		const R_T* const r_ptr, const unsigned ldr
		) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	using acc_t = typename rc_acc_type<R_T>::type;

	acc_t sum(0.), diff(0.), base(0.);

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

			const auto da = cast2acc<acc_t>(a_major == mtk::mateval::conj ? conj(a_ptr[a_index]) : a_ptr[a_index]);
			const auto db = cast2acc<acc_t>(b_major == mtk::mateval::conj ? conj(b_ptr[b_index]) : b_ptr[b_index]);
			sum = da * db + sum;
		}

		if (c_ptr != nullptr && norm2(beta) != 0) {
			std::size_t c_index;
			if (c_major == mtk::mateval::col_major) {
				c_index = row + col * ldc;
			} else {
				c_index = col + row * ldc;
			}
			const auto c = cast2acc<acc_t>(c_ptr[c_index]);
			sum = sum * alpha + c * beta;
		} else {
			sum = sum * alpha;
		}

		std::size_t r_index;
		if (r_major == mtk::mateval::col_major) {
			r_index = row + col * ldr;
		} else {
			r_index = col + row * ldr;
		}

		base = cast2acc<acc_t>(r_major == mtk::mateval::conj ? conj(r_ptr[r_index]) : r_ptr[r_index]);
		diff = sum - base;
	}

	double* my_result_ptr = result_ptr;
	if (error_type & mtk::mateval::relative_residual) {
		__shared__ double smem_diff[block_size];
		__shared__ double smem_base[block_size];
		smem_diff[threadIdx.x] = norm2(diff);
		smem_base[threadIdx.x] = norm2(base);

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

	if (error_type & mtk::mateval::max_absolute_error) {
		__shared__ double smem_error[block_size];
		smem_error[threadIdx.x] = absmax(diff);

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
		smem_element[threadIdx.x] = relative_error(diff, base);

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

	if (error_type & mtk::mateval::avg_relative_error) {
		__shared__ double smem_diff[block_size];
		smem_diff[threadIdx.x] = relative_error(diff, base);

		__syncthreads();
		for (unsigned i = block_size / 2; i >= 1; i >>= 1) {
			if (threadIdx.x < i) {
				const auto i0 = threadIdx.x;
				const auto i1 = i0 + i;
				smem_diff[i0] += smem_diff[i1];
			}
			__syncthreads();
		}
		__syncthreads();
		if (threadIdx.x == 0) {
			my_result_ptr[blockIdx.x] = smem_diff[0];
			my_result_ptr += gridDim.x;
		}
		__syncthreads();
	}
}
template <class A_T, class B_T, class C_T, class REF_T, class ACC_T>
mtk::mateval::error_map_t get_error_GEMM_core(
		const mtk::mateval::error_t error,
		const unsigned M, const unsigned N, const unsigned K,
		const mtk::mateval::layout_t a_major, const mtk::mateval::layout_t b_major, const mtk::mateval::layout_t c_major, const mtk::mateval::layout_t r_major,
		const ACC_T alpha,
		const A_T*   const a_ptr, const unsigned lda,
		const B_T*   const b_ptr, const unsigned ldb,
		const ACC_T beta,
		const C_T*   const c_ptr, const unsigned ldc,
		const REF_T* const r_ptr, const unsigned ldr
		) {
	const auto num_threads = M * N;
	const auto grid_size = (num_threads + block_size - 1) / block_size;
	using acc_t = typename rc_acc_type<REF_T>::type;

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
	if (error & mtk::mateval::avg_relative_error) {
		num_result_elements += 1;
	}

	double *h_result;
	cudaMallocHost(&h_result, grid_size * sizeof(double) * num_result_elements);
#pragma omp parallel for
	for (unsigned i = 0; i < grid_size * num_result_elements; i++) {
		h_result[i] = 0.;
	}

	error_GEMM_kernel<A_T, B_T, C_T, REF_T, acc_t, acc_t><<<grid_size, block_size>>>(
			error,
			h_result,
			M, N, K,
			a_major, b_major, c_major, r_major,
			alpha,
			a_ptr, lda,
			b_ptr, ldb,
			beta,
			c_ptr, ldc,
			r_ptr, ldr
			);
	cudaDeviceSynchronize();

	mtk::mateval::error_map_t result;

	double max_error = 0.0;
	double max_relative_error = 0.0;
	double sum_relative_error = 0.0;
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
	if (error & (mtk::mateval::max_absolute_error)) {
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
			max_relative_error = std::max(max_relative_error, tmp_result_ptr[i]);
		}
		tmp_result_ptr += grid_size;
		result.insert(std::make_pair(mtk::mateval::max_relative_error, max_relative_error));
	}
	if (error & mtk::mateval::avg_relative_error) {
#pragma omp parallel for reduction(max: sum_relative_error)
		for (unsigned i = 0; i < grid_size; i++) {
			sum_relative_error += tmp_result_ptr[i];
		}
		tmp_result_ptr += grid_size;
		result.insert(std::make_pair(mtk::mateval::avg_relative_error, sum_relative_error / (M * N)));
	}

	cudaFreeHost(h_result);

	return result;
}

} // noname namespace

template <class A_T, class B_T, class C_T, class REF_T, class ALPHA_T, class BETA_T>
mtk::mateval::error_map_t mtk::mateval::cuda::get_error_GEMM(
		const mtk::mateval::error_t error,
		const unsigned M, const unsigned N, const unsigned K,
		const mtk::mateval::layout_t a_major, const mtk::mateval::layout_t b_major, const mtk::mateval::layout_t c_major, const mtk::mateval::layout_t r_major,
		const ALPHA_T alpha,
		const A_T*   const a_ptr, const unsigned lda,
		const B_T*   const b_ptr, const unsigned ldb,
		const BETA_T beta,
		const C_T*   const c_ptr, const unsigned ldc,
		const REF_T* const r_ptr, const unsigned ldr
		) {
	using acc_t = typename rc_acc_type<REF_T>::type;

	return get_error_GEMM_core(
			error,
			M, N, K,
			a_major, b_major, c_major, r_major,
			cast2acc<acc_t>(alpha),
			a_ptr, lda,
			b_ptr, ldb,
			cast2acc<acc_t>(beta),
			c_ptr, ldc,
			r_ptr, ldr
			);
}

#define GET_ERROR_GEMM_INSTANCE_1(A_T, B_T, C_T, ALPHA_T, BETA_T) \
template mtk::mateval::error_map_t mtk::mateval::cuda::get_error_GEMM<A_T, B_T, C_T, half  , ALPHA_T, BETA_T>(const mtk::mateval::error_t, const unsigned, const unsigned, const unsigned, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const ALPHA_T, const A_T* const, const unsigned, const B_T* const, const unsigned, const BETA_T, const C_T* const, const unsigned, const half  * const, const unsigned); \
template mtk::mateval::error_map_t mtk::mateval::cuda::get_error_GEMM<A_T, B_T, C_T, float , ALPHA_T, BETA_T>(const mtk::mateval::error_t, const unsigned, const unsigned, const unsigned, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const ALPHA_T, const A_T* const, const unsigned, const B_T* const, const unsigned, const BETA_T, const C_T* const, const unsigned, const float * const, const unsigned); \
template mtk::mateval::error_map_t mtk::mateval::cuda::get_error_GEMM<A_T, B_T, C_T, double, ALPHA_T, BETA_T>(const mtk::mateval::error_t, const unsigned, const unsigned, const unsigned, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const ALPHA_T, const A_T* const, const unsigned, const B_T* const, const unsigned, const BETA_T, const C_T* const, const unsigned, const double* const, const unsigned);
#define GET_ERROR_GEMM_INSTANCE_2(A_T, B_T, ALPHA_T, BETA_T) \
	GET_ERROR_GEMM_INSTANCE_1(A_T, B_T, half  , ALPHA_T, BETA_T) \
	GET_ERROR_GEMM_INSTANCE_1(A_T, B_T, float , ALPHA_T, BETA_T) \
	GET_ERROR_GEMM_INSTANCE_1(A_T, B_T, double, ALPHA_T, BETA_T)
#define GET_ERROR_GEMM_INSTANCE_3(A_T, ALPHA_T, BETA_T) \
	GET_ERROR_GEMM_INSTANCE_2(A_T, half  , ALPHA_T, BETA_T) \
	GET_ERROR_GEMM_INSTANCE_2(A_T, float , ALPHA_T, BETA_T) \
	GET_ERROR_GEMM_INSTANCE_2(A_T, double, ALPHA_T, BETA_T)
#define GET_ERROR_GEMM_INSTANCE_4(ALPHA_T, BETA_T) \
	GET_ERROR_GEMM_INSTANCE_3(half  , ALPHA_T, BETA_T) \
	GET_ERROR_GEMM_INSTANCE_3(float , ALPHA_T, BETA_T) \
	GET_ERROR_GEMM_INSTANCE_3(double, ALPHA_T, BETA_T)
#define GET_ERROR_GEMM_INSTANCE_5(BETA_T) \
	GET_ERROR_GEMM_INSTANCE_4(half  , BETA_T) \
	GET_ERROR_GEMM_INSTANCE_4(float , BETA_T) \
	GET_ERROR_GEMM_INSTANCE_4(double, BETA_T)
#define GET_ERROR_GEMM_INSTANCE_6 \
	GET_ERROR_GEMM_INSTANCE_5(half  ) \
	GET_ERROR_GEMM_INSTANCE_5(float ) \
	GET_ERROR_GEMM_INSTANCE_5(double)
GET_ERROR_GEMM_INSTANCE_6

#define GET_ERROR_COMPLEX_GEMM_INSTANCE_1(A_T, B_T, C_T, ALPHA_T, BETA_T) \
template mtk::mateval::error_map_t mtk::mateval::cuda::get_error_GEMM<A_T, B_T, C_T, cuComplex      , ALPHA_T, BETA_T>(const mtk::mateval::error_t, const unsigned, const unsigned, const unsigned, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const ALPHA_T, const A_T* const, const unsigned, const B_T* const, const unsigned, const BETA_T, const C_T* const, const unsigned, const cuComplex      * const, const unsigned); \
template mtk::mateval::error_map_t mtk::mateval::cuda::get_error_GEMM<A_T, B_T, C_T, cuDoubleComplex, ALPHA_T, BETA_T>(const mtk::mateval::error_t, const unsigned, const unsigned, const unsigned, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const ALPHA_T, const A_T* const, const unsigned, const B_T* const, const unsigned, const BETA_T, const C_T* const, const unsigned, const cuDoubleComplex* const, const unsigned);
#define GET_ERROR_COMPLEX_GEMM_INSTANCE_2(A_T, B_T, ALPHA_T, BETA_T) \
	GET_ERROR_COMPLEX_GEMM_INSTANCE_1(A_T, B_T, cuComplex      , ALPHA_T, BETA_T) \
	GET_ERROR_COMPLEX_GEMM_INSTANCE_1(A_T, B_T, cuDoubleComplex, ALPHA_T, BETA_T)
#define GET_ERROR_COMPLEX_GEMM_INSTANCE_3(A_T, ALPHA_T, BETA_T) \
	GET_ERROR_COMPLEX_GEMM_INSTANCE_2(A_T, cuComplex      , ALPHA_T, BETA_T) \
	GET_ERROR_COMPLEX_GEMM_INSTANCE_2(A_T, cuDoubleComplex, ALPHA_T, BETA_T)
#define GET_ERROR_COMPLEX_GEMM_INSTANCE_4(ALPHA_T, BETA_T) \
	GET_ERROR_COMPLEX_GEMM_INSTANCE_3(cuComplex      , ALPHA_T, BETA_T) \
	GET_ERROR_COMPLEX_GEMM_INSTANCE_3(cuDoubleComplex, ALPHA_T, BETA_T)
#define GET_ERROR_COMPLEX_GEMM_INSTANCE_5(BETA_T) \
	GET_ERROR_COMPLEX_GEMM_INSTANCE_4(cuComplex      , BETA_T) \
	GET_ERROR_COMPLEX_GEMM_INSTANCE_4(cuDoubleComplex, BETA_T)
#define GET_ERROR_COMPLEX_GEMM_INSTANCE_6 \
	GET_ERROR_COMPLEX_GEMM_INSTANCE_5(cuComplex      ) \
	GET_ERROR_COMPLEX_GEMM_INSTANCE_5(cuDoubleComplex)
GET_ERROR_COMPLEX_GEMM_INSTANCE_6

template <class A_T, class B_T, class REF_T>
mtk::mateval::error_map_t mtk::mateval::cuda::get_error_AxB(
		const mtk::mateval::error_t error,
		const unsigned M, const unsigned N, const unsigned K,
		const mtk::mateval::layout_t a_major, const mtk::mateval::layout_t b_major, const mtk::mateval::layout_t r_major,
		const A_T*   const a_ptr, const unsigned lda,
		const B_T*   const b_ptr, const unsigned ldb,
		const REF_T* const r_ptr, const unsigned ldr
		) {
	using acc_t = typename rc_acc_type<REF_T>::type;

	return get_error_GEMM_core(
			error,
			M, N, K,
			a_major, b_major, mtk::mateval::col_major,r_major,
			acc_t(1.),
			a_ptr, lda,
			b_ptr, ldb,
			acc_t(0.),
			reinterpret_cast<REF_T*>(0), 0,
			r_ptr, ldr
			);
}

#define GET_ERROR_AXB_INSTANCE_1(A_T, B_T) \
template mtk::mateval::error_map_t mtk::mateval::cuda::get_error_AxB<A_T, B_T, half  >(const mtk::mateval::error_t, const unsigned, const unsigned, const unsigned, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const A_T* const, const unsigned, const B_T* const, const unsigned, const half  * const, const unsigned); \
template mtk::mateval::error_map_t mtk::mateval::cuda::get_error_AxB<A_T, B_T, float >(const mtk::mateval::error_t, const unsigned, const unsigned, const unsigned, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const A_T* const, const unsigned, const B_T* const, const unsigned, const float * const, const unsigned); \
template mtk::mateval::error_map_t mtk::mateval::cuda::get_error_AxB<A_T, B_T, double>(const mtk::mateval::error_t, const unsigned, const unsigned, const unsigned, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const A_T* const, const unsigned, const B_T* const, const unsigned, const double* const, const unsigned);
#define GET_ERROR_AXB_INSTANCE_2(A_T) \
	GET_ERROR_AXB_INSTANCE_1(A_T, half) \
	GET_ERROR_AXB_INSTANCE_1(A_T, float) \
	GET_ERROR_AXB_INSTANCE_1(A_T, double)
#define GET_ERROR_AXB_INSTANCE_3 \
	GET_ERROR_AXB_INSTANCE_2(half) \
	GET_ERROR_AXB_INSTANCE_2(float) \
	GET_ERROR_AXB_INSTANCE_2(double)
GET_ERROR_AXB_INSTANCE_3

#define GET_ERROR_AXB_INSTANCE_1_COMPLEX(A_T, B_T) \
template mtk::mateval::error_map_t mtk::mateval::cuda::get_error_AxB<A_T, B_T, cuComplex >(const mtk::mateval::error_t, const unsigned, const unsigned, const unsigned, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const A_T* const, const unsigned, const B_T* const, const unsigned, const cuComplex * const, const unsigned); \
template mtk::mateval::error_map_t mtk::mateval::cuda::get_error_AxB<A_T, B_T, cuDoubleComplex>(const mtk::mateval::error_t, const unsigned, const unsigned, const unsigned, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const A_T* const, const unsigned, const B_T* const, const unsigned, const cuDoubleComplex* const, const unsigned);
#define GET_ERROR_AXB_INSTANCE_2_COMPLEX(A_T) \
	GET_ERROR_AXB_INSTANCE_1_COMPLEX(A_T, cuComplex) \
	GET_ERROR_AXB_INSTANCE_1_COMPLEX(A_T, cuDoubleComplex)
#define GET_ERROR_AXB_INSTANCE_3_COMPLEX \
	GET_ERROR_AXB_INSTANCE_2_COMPLEX(cuComplex) \
	GET_ERROR_AXB_INSTANCE_2_COMPLEX(cuDoubleComplex)
GET_ERROR_AXB_INSTANCE_3_COMPLEX

namespace {
template <class A_T, class R_T>
__global__ void error_kernel(
		const mtk::mateval::error_t error_type,
		double* const result_ptr,
		const unsigned M, const unsigned N,
		const mtk::mateval::layout_t a_major, const mtk::mateval::layout_t r_major,
		const A_T* const a_ptr, const unsigned lda,
		const R_T* const r_ptr, const unsigned ldr
		) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

	using acc_t = typename rc_acc_type<R_T>::type;

	acc_t sum(0.), diff(0.), base(0.);

	if (tid < M * N) {
		const auto row = tid % M;
		const auto col = tid / M;

		std::size_t a_index;
		if (a_major == mtk::mateval::col_major) {
			a_index = row + col * lda;
		} else {
			a_index = col + row * lda;
		}

		sum = static_cast<double>(a_major == mtk::mateval::conj ? conj(a_ptr[a_index]) : a_ptr[a_index]);

		std::size_t r_index;
		if (r_major == mtk::mateval::col_major) {
			r_index = row + col * ldr;
		} else {
			r_index = col + row * ldr;
		}

		base = a_major == mtk::mateval::conj ? conj(r_ptr[r_index]) : r_ptr[r_index];
		diff = sum - base;
	}

	double* my_result_ptr = result_ptr;
	if (error_type & mtk::mateval::relative_residual) {
		__shared__ double smem_diff[block_size];
		__shared__ double smem_base[block_size];
		smem_diff[threadIdx.x] = norm2(diff);
		smem_base[threadIdx.x] = norm2(base);

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

	if (error_type & mtk::mateval::max_absolute_error) {
		__shared__ double smem_error[block_size];
		smem_error[threadIdx.x] = absmax(diff);

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
		smem_element[threadIdx.x] = (norm2(base) != 0) ? sqrt(norm2(diff) / norm2(base)) : 0;

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

	if (error_type & mtk::mateval::avg_relative_error) {
		__shared__ double smem_diff[block_size];
		smem_diff[threadIdx.x] = (absmax(base) != 0) ? sqrt(norm2(diff) / norm2(base)) : 0;

		__syncthreads();
		for (unsigned i = block_size / 2; i >= 1; i >>= 1) {
			if (threadIdx.x < i) {
				const auto i0 = threadIdx.x;
				const auto i1 = i0 + i;
				smem_diff[i0] += smem_diff[i1];
			}
			__syncthreads();
		}
		__syncthreads();
		if (threadIdx.x == 0) {
			my_result_ptr[blockIdx.x] = smem_diff[0];
			my_result_ptr += gridDim.x;
		}
		__syncthreads();
	}
}
}

template <class A_T, class REF_T>
mtk::mateval::error_map_t mtk::mateval::cuda::get_error(
		const mtk::mateval::error_t error,
		const unsigned M, const unsigned N,
		const mtk::mateval::layout_t a_major, const mtk::mateval::layout_t r_major,
		const A_T*   const a_ptr, const unsigned lda,
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
	if (error & mtk::mateval::avg_relative_error) {
		num_result_elements += 1;
	}

	double *h_result;
	cudaMallocHost(&h_result, grid_size * sizeof(double) * num_result_elements);
#pragma omp parallel for
	for (unsigned i = 0; i < grid_size * num_result_elements; i++) {
		h_result[i] = 0.;
	}

	error_kernel<A_T, REF_T><<<grid_size, block_size>>>(
			error,
			h_result,
			M, N,
			a_major, r_major,
			a_ptr, lda,
			r_ptr, ldr
			);
	cudaDeviceSynchronize();

	mtk::mateval::error_map_t result;

	double max_error = 0.0;
	double sum_error = 0.0;
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
			max_error = std::max(max_error, tmp_result_ptr[i]);
		}
		tmp_result_ptr += grid_size;
		result.insert(std::make_pair(mtk::mateval::max_relative_error, max_error));
	}
	if (error & mtk::mateval::avg_relative_error) {
#pragma omp parallel for reduction(max: max_element)
		for (unsigned i = 0; i < grid_size; i++) {
			sum_error += tmp_result_ptr[i];
		}
		tmp_result_ptr += grid_size;
		result.insert(std::make_pair(mtk::mateval::avg_relative_error, sum_error / (M * N)));
	}

	cudaFreeHost(h_result);

	return result;
}

template mtk::mateval::error_map_t mtk::mateval::cuda::get_error<half  , half  >(const mtk::mateval::error_t, const unsigned, const unsigned, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const half  * const, const unsigned, const half  * const, const unsigned);
template mtk::mateval::error_map_t mtk::mateval::cuda::get_error<half  , float >(const mtk::mateval::error_t, const unsigned, const unsigned, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const half  * const, const unsigned, const float * const, const unsigned);
template mtk::mateval::error_map_t mtk::mateval::cuda::get_error<half  , double>(const mtk::mateval::error_t, const unsigned, const unsigned, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const half  * const, const unsigned, const double* const, const unsigned);
template mtk::mateval::error_map_t mtk::mateval::cuda::get_error<float , half  >(const mtk::mateval::error_t, const unsigned, const unsigned, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const float * const, const unsigned, const half  * const, const unsigned);
template mtk::mateval::error_map_t mtk::mateval::cuda::get_error<float , float >(const mtk::mateval::error_t, const unsigned, const unsigned, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const float * const, const unsigned, const float * const, const unsigned);
template mtk::mateval::error_map_t mtk::mateval::cuda::get_error<float , double>(const mtk::mateval::error_t, const unsigned, const unsigned, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const float * const, const unsigned, const double* const, const unsigned);
template mtk::mateval::error_map_t mtk::mateval::cuda::get_error<double, half  >(const mtk::mateval::error_t, const unsigned, const unsigned, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const double* const, const unsigned, const half  * const, const unsigned);
template mtk::mateval::error_map_t mtk::mateval::cuda::get_error<double, float >(const mtk::mateval::error_t, const unsigned, const unsigned, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const double* const, const unsigned, const float * const, const unsigned);
template mtk::mateval::error_map_t mtk::mateval::cuda::get_error<double, double>(const mtk::mateval::error_t, const unsigned, const unsigned, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const double* const, const unsigned, const double* const, const unsigned);

namespace {
// SVD
template <class U_T, class S_T, class V_T, class REF_T>
__global__ void residual_SVD_kernel(
		double* const diff_norm,
		double* const base_norm,
		const unsigned M, const unsigned N, const unsigned K,
		const mtk::mateval::layout_t u_major, const mtk::mateval::layout_t v_major, const mtk::mateval::layout_t r_major,
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
}

template <class U_T, class S_T, class V_T, class REF_T>
double mtk::mateval::cuda::residual_UxSxVt(
		const unsigned M, const unsigned N, const unsigned K,
		const mtk::mateval::layout_t u_major, const mtk::mateval::layout_t v_major, const mtk::mateval::layout_t r_major,
		const U_T*   const u_ptr, const unsigned ldu,
		const S_T*   const s_ptr,
		const V_T*   const v_ptr, const unsigned ldv,
		const REF_T* const r_ptr, const unsigned ldr
		) {
	if (u_major == mtk::mateval::conj) throw std::runtime_error("`conj` is not supported for U");
	if (v_major == mtk::mateval::conj) throw std::runtime_error("`conj` is not supported for V");
	if (r_major == mtk::mateval::conj) throw std::runtime_error("`conj` is not supported for R");
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

template double mtk::mateval::cuda::residual_UxSxVt<double, double, double, double>(const unsigned, const unsigned, const unsigned, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const double* const, const unsigned, const double* const, const double* const, const unsigned, const double* const, const unsigned);
template double mtk::mateval::cuda::residual_UxSxVt<float , float , float , float >(const unsigned, const unsigned, const unsigned, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const float * const, const unsigned, const float * const, const float * const, const unsigned, const float * const, const unsigned);
template double mtk::mateval::cuda::residual_UxSxVt<half  , half  , half  , half  >(const unsigned, const unsigned, const unsigned, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const mtk::mateval::layout_t, const half  * const, const unsigned, const half  * const, const half  * const, const unsigned, const half  * const, const unsigned);

template <class T>
__global__ void orthogonality_kernel(
		double* const diff_norm,
		const unsigned M, const unsigned N,
		const mtk::mateval::layout_t major,
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
		const mtk::mateval::layout_t major,
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

template double mtk::mateval::cuda::orthogonality<double>(const unsigned, const unsigned, const mtk::mateval::layout_t, const double* const, const unsigned);
template double mtk::mateval::cuda::orthogonality<float >(const unsigned, const unsigned, const mtk::mateval::layout_t, const float * const, const unsigned);
template double mtk::mateval::cuda::orthogonality<half  >(const unsigned, const unsigned, const mtk::mateval::layout_t, const half  * const, const unsigned);
