#include <iostream>
#include <memory>
#include <cublas.h>
#include <mateval/comparison_cuda.hpp>

const std::size_t matrix_dim = 1000;
const std::size_t matrix_ld  = 1200;

void test_AxB(
	const unsigned M,
	const unsigned N,
	const unsigned K,
	const mtk::mateval::major_t a_major,
	const mtk::mateval::major_t b_major,
	const mtk::mateval::major_t r_major,
	const unsigned lda,
	const unsigned ldb,
	const unsigned ldr,
	const bool should_be_passed
	) {
	const std::size_t a_mem_size = lda * (a_major == mtk::mateval::col_major ? K : M);
	const std::size_t b_mem_size = ldb * (b_major == mtk::mateval::col_major ? N : K);
	const std::size_t r_mem_size = ldr * (r_major == mtk::mateval::col_major ? N : M);

	auto mat_a = std::unique_ptr<float[]>(new float [a_mem_size]);
	auto mat_b = std::unique_ptr<float[]>(new float [b_mem_size]);
	auto mat_r = std::unique_ptr<float[]>(new float [r_mem_size]);

	float *da, *db, *dr;
	cudaMalloc(&da, sizeof(float) * a_mem_size);
	cudaMalloc(&db, sizeof(float) * b_mem_size);
	cudaMalloc(&dr, sizeof(float) * r_mem_size);

	// Set A
	for (unsigned m = 0; m < M; m++) {
		for (unsigned n = 0; n < K; n++) {
			const auto index = (a_major == mtk::mateval::col_major ? (m + n * lda) : (m * lda + n));
			mat_a.get()[index] = (n + 1) * (m + 1) / static_cast<float>(M);
		}
	}
	// Set B
	for (unsigned m = 0; m < K; m++) {
		for (unsigned n = 0; n < N; n++) {
			const auto index = (b_major == mtk::mateval::col_major ? (m + n * ldb) : (m * ldb + n));
			mat_b.get()[index] = (m + 1) * (n + 1) / static_cast<float>(N);
		}
	}
	// Set ref
	for (unsigned m = 0; m < M; m++) {
		for (unsigned n = 0; n < N; n++) {
			const auto index = (r_major == mtk::mateval::col_major ? (m + n * ldr) : (m * ldr + n));
			mat_r.get()[index] = static_cast<double>(K * (K + 1) * (2 * K + 1) / 6) * (m + 1) * (n + 1) / static_cast<double>(M * N);
			if (!should_be_passed) {
				mat_r.get()[index] *= -1.0f;
			}
		}
	}

	cudaMemcpy(da, mat_a.get(), sizeof(float) * a_mem_size, cudaMemcpyDefault);
	cudaMemcpy(db, mat_b.get(), sizeof(float) * b_mem_size, cudaMemcpyDefault);
	cudaMemcpy(dr, mat_r.get(), sizeof(float) * r_mem_size, cudaMemcpyDefault);

	const auto residual = mtk::mateval::cuda::residual_AxB(
		M, N, K,
		a_major, b_major, r_major,
		da, lda,
		db, ldb,
		dr, ldr
		);

	const auto max_error = mtk::mateval::cuda::max_error_AxB(
		M, N, K,
		a_major, b_major, r_major,
		da, lda,
		db, ldb,
		dr, ldr
		);

	const auto max_relative_error = mtk::mateval::cuda::max_relative_error_AxB(
		M, N, K,
		a_major, b_major, r_major,
		da, lda,
		db, ldb,
		dr, ldr
		);
	std::printf("[%s]{M=%3u,N=%3u,K=%u,lda=%u,ldb=%u,ldr=%u,a_major=%3s,b_major=%3s,r_major=%3s} residual=%e(%6s), max_error=%e(%6s), max_relative_error=%e(%6s)\n",
				__func__,
				M, N, K,
				lda, ldb, ldr,
				(a_major == mtk::mateval::col_major ? "col" : "row"),
				(b_major == mtk::mateval::col_major ? "col" : "row"),
				(r_major == mtk::mateval::col_major ? "col" : "row"),
				residual,
				((residual < 1e-6) == should_be_passed ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m"),
				max_error,
				((max_error < (K * K * K * 5e-8) == should_be_passed) ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m"),
				max_relative_error,
				((max_relative_error < (1e-6) == should_be_passed) ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m")
				);

	cudaFree(da);
	cudaFree(db);
}

int main() {
	std::printf("----------- passing test -----------\n");
	test_AxB(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_AxB(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	std::printf("--------- failing test ---------\n");
	test_AxB(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_AxB(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
}
