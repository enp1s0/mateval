#include <iostream>
#include <mateval/latms_cuda.hpp>
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
	const auto cond_ref = s[0] / s[rank - 1];

	mtk::mateval::cuda::latms(
			A,
			m, n,
			s,
			rank,
			0, // seed
			memory_type
			);

	cudaDeviceSynchronize();

	const auto cond = mtk::mateval::cond(
			m, n, mtk::mateval::col_major,
			A, m);

	std::printf("[%s] A size=(%5u, %5u), type=%6s, mem=%6s, rank=%5u, ref cond=%e, actual cond=%e\n",
			__func__,
			m, n,
			get_name<T>(),
			get_memory_type_name(memory_type),
			rank,
			cond_ref,
			cond
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
