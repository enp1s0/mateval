#include <iostream>
#include <random>
#include <cuda_fp16.h>
#include <mateval/cuda/minmax.hpp>

template <class T>
const char* get_str();
template <> const char* get_str<half  >() {return "half";}
template <> const char* get_str<float >() {return "float";}
template <> const char* get_str<double>() {return "double";}

template <class T>
void test(const std::size_t N) {
	T* mat_ptr;
	cudaMallocManaged(&mat_ptr, sizeof(T) * N * N);

	std::mt19937 mt(0);
	std::uniform_real_distribution<double> dist(-1, 1);

	for (std::size_t i = 0; i < N * N; i++) {
		mat_ptr[i] = dist(mt);
	}

	const auto res = mtk::mateval::cuda::operate(
		mtk::mateval::op_abs_max | mtk::mateval::op_abs_min | mtk::mateval::op_max | mtk::mateval::op_min,
		mtk::mateval::col_major,
		N, N,
		mat_ptr,
		N
		);

	std::printf("[%7s, N=%lu] op_min = %e, op_max = %e, op_abs_min = %e, op_abs_max = %e\n",
							get_str<T>(),
							N,
							res.at(mtk::mateval::op_min),
							res.at(mtk::mateval::op_max),
							res.at(mtk::mateval::op_abs_min),
							res.at(mtk::mateval::op_abs_max)
							);

	cudaFree(mat_ptr);
}

int main() {
	const std::size_t N = 2000;

	test<double>(N);
	test<float >(N);
	test<half  >(N);
}
