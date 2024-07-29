#include <mateval/cuda/latms.hpp>
#include <sstream>
#include <curand.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#define MATEVAL_CUSOLVER_CHECK_ERROR(status) check_cusolver_error((status), __FILE__, __LINE__, __func__)
namespace {
void check_cusolver_error(
		const cusolverStatus_t cusolver_status,
		const std::string file,
		const std::size_t line,
		const std::string func) {
	if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
		std::string error_string;
#define CUSOLVER_ERROR_CASE(c) case c: error_string = #c; break
		switch(cusolver_status){
			CUSOLVER_ERROR_CASE( CUSOLVER_STATUS_NOT_INITIALIZED );
			CUSOLVER_ERROR_CASE( CUSOLVER_STATUS_ALLOC_FAILED );
			CUSOLVER_ERROR_CASE( CUSOLVER_STATUS_INVALID_VALUE );
			CUSOLVER_ERROR_CASE( CUSOLVER_STATUS_ARCH_MISMATCH );
			CUSOLVER_ERROR_CASE( CUSOLVER_STATUS_EXECUTION_FAILED );
			CUSOLVER_ERROR_CASE( CUSOLVER_STATUS_INTERNAL_ERROR );
			CUSOLVER_ERROR_CASE( CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED );
		default: error_string = "Unknown error"; break;
		}
		std::stringstream ss;
		ss<< error_string;
		ss<<" ["<<file<<":"<<line<<" in "<<func<<"]";
		throw std::runtime_error(ss.str());
	}
}
}

namespace {
curandStatus_t curandGenerateUniform_wrapper(
		curandGenerator_t curand_generator, float* const ptr, const std::size_t size
		) {
	return curandGenerateUniform(curand_generator, ptr, size);
}
curandStatus_t curandGenerateUniform_wrapper(
		curandGenerator_t curand_generator, double* const ptr, const std::size_t size
		) {
	return curandGenerateUniformDouble(curand_generator, ptr, size);
}

cusolverStatus_t cusolverDnXorgqr_bufferSize_wrapper(
		cusolverDnHandle_t handle, int m, int n, int k, const float *A, int lda, const float *tau, int *lwork
		) {
	return cusolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
}
cusolverStatus_t cusolverDnXorgqr_bufferSize_wrapper(
		cusolverDnHandle_t handle, int m, int n, int k, const double *A, int lda, const double *tau, int *lwork
		) {
	return cusolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
}

cusolverStatus_t cusolverDnXorgqr_wrapper(
		cusolverDnHandle_t handle, int m, int n, int k, float *A, int lda, const float *tau, float* work, int lwork, int *devInfo
		) {
	return cusolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
}
cusolverStatus_t cusolverDnXorgqr_wrapper(
		cusolverDnHandle_t handle, int m, int n, int k, double *A, int lda, const double *tau, double* work, int lwork, int *devInfo
		) {
	return cusolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
}

template <class T>
cudaDataType getCudaDataType();
template <>
cudaDataType getCudaDataType<float >() {return CUDA_R_32F;};
template <>
cudaDataType getCudaDataType<double>() {return CUDA_R_64F;};

template <class D, class S>
__device__ D convert(const S a) {return a;};

constexpr unsigned block_size = 256;

template <class OUTPUT_T, class S_T, class TMP_T>
__global__ void multiply_usvt(
		const unsigned m, const unsigned n,
		mtk::mateval::layout_t major,
		OUTPUT_T* const out_ptr,
		const unsigned ldm,
		const S_T *s_ptr,
		const unsigned rank,
		const TMP_T* u_ptr,
		const TMP_T* v_ptr
		) {
	extern __shared__ uint32_t smem[];
	auto smem_s = reinterpret_cast<S_T*>(smem);
	// Copy s to smem from gmem
	for (unsigned i = 0; i < rank; i += blockDim.x) {
		const auto index = i + threadIdx.x;
		if (index >= rank) break;
		smem_s[index] = s_ptr[index];
	}
	__syncthreads();

	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= m * n) {
		return;
	}


	const auto i = tid % m;
	const auto j = tid / m;

	std::size_t gmem_index = 0;
	if (major == mtk::mateval::col_major) {
		gmem_index = i + j * ldm;
	} else {
		gmem_index = j + i * ldm;
	}

	TMP_T v = 0;
	for (unsigned r = 0; r < rank; r++) {
		v += u_ptr[r * m + i] * smem_s[r] * v_ptr[r * n + j];
	}
	out_ptr[gmem_index] = convert<OUTPUT_T>(v);
}
} // noname namespace

template <class T>
void mtk::mateval::cuda::latms(
		const unsigned m,
		const unsigned n,
		const mtk::mateval::layout_t major,
		T* const dst_ptr,
		const unsigned ldm,
		T* const d,
		const unsigned rank,
		const unsigned long long seed,
		const memory_type working_memory_type,
		cudaStream_t cuda_stream
		) {
	// Allocate working memory
	T *mat_u, *mat_v, *diag_s, *tau;
	if (working_memory_type == mtk::mateval::cuda::device_memory) {
		cudaMalloc(&mat_u , sizeof(T) * m * rank);
		cudaMalloc(&mat_v , sizeof(T) * n * rank);
		cudaMalloc(&diag_s, sizeof(T) * rank);
		cudaMalloc(&tau, sizeof(T) * std::max(std::min(rank, m), std::min(rank, n)));

		cudaMemcpy(diag_s, d, sizeof(T) * rank, cudaMemcpyDefault);
	} else {
		cudaMallocHost(&mat_u , sizeof(T) * m * rank);
		cudaMallocHost(&mat_v , sizeof(T) * n * rank);
		cudaMallocHost(&diag_s, sizeof(T) * rank);
		cudaMallocHost(&tau, sizeof(T) * std::max(std::min(rank, m), std::min(rank, n)));

		for (unsigned i = 0; i < rank; i++) {
			diag_s[i] = d[i];
		}
	}
	// Copy S
	// Init U, V
	curandGenerator_t curand_generator;
	curandCreateGenerator(&curand_generator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetStream(curand_generator, cuda_stream);
	curandSetPseudoRandomGeneratorSeed(curand_generator, seed);

	curandGenerateUniform_wrapper(curand_generator, mat_u, m * rank);
	curandGenerateUniform_wrapper(curand_generator, mat_v, rank * n);

	// Destroy generator
	curandDestroyGenerator(curand_generator);

	// geqrf & ormqr
	cusolverDnHandle_t cusolver_handle;
	MATEVAL_CUSOLVER_CHECK_ERROR(cusolverDnCreate(&cusolver_handle));
	MATEVAL_CUSOLVER_CHECK_ERROR(cusolverDnSetStream(cusolver_handle, cuda_stream));
	cusolverDnParams_t cusolver_params;
	MATEVAL_CUSOLVER_CHECK_ERROR(cusolverDnCreateParams(&cusolver_params));

	// get buffer size
	std::size_t geqrf_lwork_u_device, geqrf_lwork_v_device;
	std::size_t geqrf_lwork_u_host, geqrf_lwork_v_host;
	int orgqr_lwork_u, orgqr_lwork_v;
	MATEVAL_CUSOLVER_CHECK_ERROR(cusolverDnXgeqrf_bufferSize(cusolver_handle, cusolver_params, m, rank,
			getCudaDataType<T>(), mat_u, m, getCudaDataType<T>(), tau, getCudaDataType<T>(), &geqrf_lwork_u_device, &geqrf_lwork_u_host));
	MATEVAL_CUSOLVER_CHECK_ERROR(cusolverDnXgeqrf_bufferSize(cusolver_handle, cusolver_params, n, rank,
			getCudaDataType<T>(), mat_v, n, getCudaDataType<T>(), tau, getCudaDataType<T>(), &geqrf_lwork_v_device, &geqrf_lwork_v_host));
	MATEVAL_CUSOLVER_CHECK_ERROR(cusolverDnXorgqr_bufferSize_wrapper(cusolver_handle, m, rank, rank, mat_u, m, tau, &orgqr_lwork_u));
	MATEVAL_CUSOLVER_CHECK_ERROR(cusolverDnXorgqr_bufferSize_wrapper(cusolver_handle, n, rank, rank, mat_v, n, tau, &orgqr_lwork_v));
	const auto max_lwork = std::max(orgqr_lwork_u, orgqr_lwork_v);

	T* qr_work;
	int* qr_info;
	if (working_memory_type == mtk::mateval::cuda::device_memory) {
		cudaMalloc(&qr_work, sizeof(T) * max_lwork);
		cudaMalloc(&qr_info, sizeof(int));
	} else {
		cudaMallocHost(&qr_work, sizeof(T) * max_lwork);
		cudaMallocHost(&qr_info, sizeof(int));
	}

	T* qr_work_host;
	T* qr_work_device;
	cudaMalloc(&qr_work_device, sizeof(T) * std::max(geqrf_lwork_u_device, geqrf_lwork_v_device));
	cudaMallocHost(&qr_work_host, sizeof(T) * std::max(geqrf_lwork_u_host, geqrf_lwork_v_host));

	// Orthogonalize U and V
	MATEVAL_CUSOLVER_CHECK_ERROR(cusolverDnXgeqrf(
			cusolver_handle, cusolver_params, m, rank,
			getCudaDataType<T>(), mat_u, m, getCudaDataType<T>(), tau, getCudaDataType<T>(),
			qr_work_device, geqrf_lwork_u_device,
			qr_work_host, geqrf_lwork_u_host,
			qr_info));
	MATEVAL_CUSOLVER_CHECK_ERROR(cusolverDnXorgqr_wrapper(
			cusolver_handle, m, rank, rank, mat_u, m, tau, qr_work, orgqr_lwork_u, qr_info));
	MATEVAL_CUSOLVER_CHECK_ERROR(cusolverDnXgeqrf(
			cusolver_handle, cusolver_params, n, rank,
			getCudaDataType<T>(), mat_v, n, getCudaDataType<T>(), tau, getCudaDataType<T>(),
			qr_work_device, geqrf_lwork_v_device,
			qr_work_host, geqrf_lwork_v_host,
			qr_info));
	MATEVAL_CUSOLVER_CHECK_ERROR(cusolverDnXorgqr_wrapper(
			cusolver_handle, n, rank, rank, mat_v, n, tau, qr_work, orgqr_lwork_v, qr_info));

	// Destroy handlers
	MATEVAL_CUSOLVER_CHECK_ERROR(cusolverDnDestroyParams(cusolver_params));
	MATEVAL_CUSOLVER_CHECK_ERROR(cusolverDnDestroy(cusolver_handle));

	// Free working memory
	if (working_memory_type == mtk::mateval::cuda::device_memory) {
		cudaFree(qr_work);
		cudaFree(qr_info);
	} else {
		cudaFreeHost(qr_work);
		cudaFreeHost(qr_info);
	}
	cudaFree(qr_work_device);
	cudaFreeHost(qr_work_host);

	// Multiply U s Vt
	const auto shared_memory_size = sizeof(T) * rank;
	cudaFuncSetAttribute(&multiply_usvt<T, T, T>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size);

	multiply_usvt<T, T, T><<<(m * n + block_size - 1) / block_size, block_size, shared_memory_size, cuda_stream>>>(
			m, n,
			major,
			dst_ptr,
			ldm,
			diag_s,
			rank,
			mat_u, mat_v
			);

	if (working_memory_type == mtk::mateval::cuda::device_memory) {
		cudaFree(mat_u);
		cudaFree(mat_v);
		cudaFree(diag_s);
		cudaFree(tau);
	} else {
		cudaFreeHost(mat_u);
		cudaFreeHost(mat_v);
		cudaFreeHost(diag_s);
		cudaFreeHost(tau);
	}
}

template void mtk::mateval::cuda::latms<float >(const unsigned, const unsigned, mtk::mateval::layout_t,  float* const, const unsigned, float * const, const unsigned, const unsigned long long, const mtk::mateval::cuda::memory_type, cudaStream_t);
template void mtk::mateval::cuda::latms<double>(const unsigned, const unsigned, mtk::mateval::layout_t, double* const, const unsigned, double* const, const unsigned, const unsigned long long, const mtk::mateval::cuda::memory_type, cudaStream_t);
