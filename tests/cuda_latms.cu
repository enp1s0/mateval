#include <iostream>
#include <mateval/cuda/latms.hpp>
#include <mateval/cond.hpp>

namespace {

template <class T>
const char* get_name();
template <> const char* get_name<float >() {return "float" ;}
template <> const char* get_name<double>() {return "double";}

const char* get_memory_type_name(const mtk::mateval::cuda::memory_type t) {
	switch (t) {
	case mtk::mateval::cuda::device_memory:
		return "device";
	case mtk::mateval::cuda::host_memory:
		return "host";
	default:
		return "unknown";
	}
}

void svd(int matrix_layout, char jobu, char jobvt, lapack_int m, lapack_int n, double* a, lapack_int lda, double* s, double* u, lapack_int ldu, double* vt, lapack_int ldvt, double* work, lapack_int lwork) {
	LAPACKE_dgesvd_work(matrix_layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork);
}

void svd(int matrix_layout, char jobu, char jobvt, lapack_int m, lapack_int n, float* a, lapack_int lda, float* s, float* u, lapack_int ldu, float* vt, lapack_int ldvt, float* work, lapack_int lwork) {
	LAPACKE_sgesvd_work(matrix_layout, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork);
}

template <class T>
void get_singular_value(
		T* const s_ptr,
		const std::size_t m, const std::size_t n,
		T* const a_ptr, const std::size_t lda
		) {
	int lwork = -1;
	T* work = nullptr;
	T tmp;
	svd(LAPACK_COL_MAJOR, 'N', 'N', m, n, a_ptr, lda, s_ptr, nullptr, 1, nullptr, 1, &tmp, lwork);

	lwork = static_cast<int>(tmp);
	work = new T [lwork];
	svd(LAPACK_COL_MAJOR, 'N', 'N', m, n, a_ptr, lda, s_ptr, nullptr, 1, nullptr, 1, work, lwork);

	delete [] work;
}

template <class T>
void test_cuda_latms(
		const unsigned m, const unsigned n,
		const unsigned rank,
		const mtk::mateval::cuda::memory_type memory_type
		) {
	T* A;
	cudaMallocHost(&A, m * n * sizeof(T));

	T* s;
	cudaMallocHost(&s, rank * sizeof(T));
	for (unsigned i = 0; i < rank; i++) {
		s[i] = (rank - i) / static_cast<T>(rank) + 1;
	}

	mtk::mateval::cuda::latms(
			m, n,
			mtk::mateval::col_major,
			A, m,
			s,
			rank,
			0, // seed
			memory_type
			);

	cudaDeviceSynchronize();

	std::vector<T> s_computed(std::min(m, n));
	get_singular_value(s_computed.data(), m, n, A, m);

	double error = 0.0f;
	for (unsigned i = 0; i < rank; i++) {
		const auto diff = s_computed[i] - s[i];
		error += diff * diff;
	}

	std::printf("[%s] A size=(%5u, %5u), type=%6s, mem=%6s, rank=%5u, error=%e\n",
			__func__,
			m, n,
			get_name<T>(),
			get_memory_type_name(memory_type),
			rank,
			std::sqrt(error)
			);

	cudaFreeHost(A);
	cudaFreeHost(s);
}
} // noname namespace

int main() {
	for (unsigned log_m = 5; log_m <= 10; log_m++) {
		for (unsigned log_n = 5; log_n <= 10; log_n++) {
			const auto m = 1u << log_m;
			const auto n = 1u << log_n;
			for (unsigned rank = 1; rank <= std::min(m, n); rank <<= 1) {
				test_cuda_latms<float >(m, n, rank, mtk::mateval::cuda::device_memory);
				test_cuda_latms<float >(m, n, rank, mtk::mateval::cuda::host_memory  );
				test_cuda_latms<double>(m, n, rank, mtk::mateval::cuda::device_memory);
				test_cuda_latms<double>(m, n, rank, mtk::mateval::cuda::host_memory  );
			}
		}
	}
}
