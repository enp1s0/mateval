#include <iostream>
#include <memory>
#include <mateval/mateval.hpp>

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

	const auto residual = mtk::mateval::residual_AxB(
		M, N, K,
		a_major, b_major, r_major,
		mat_a.get(), lda,
		mat_b.get(), ldb,
		mat_r.get(), ldr
		);

	const auto max_error = mtk::mateval::max_error_AxB(
		M, N, K,
		a_major, b_major, r_major,
		mat_a.get(), lda,
		mat_b.get(), ldb,
		mat_r.get(), ldr
		);
	std::printf("[test]{M=%3u,N=%3u,K=%u,lda=%u,ldb=%u,ldr=%u,a_major=%3s,b_major=%3s,r_major=%3s} residual=%e(%6s), max_error=%e(%6s)\n",
				M, N, K,
				lda, ldb, ldr,
				(a_major == mtk::mateval::col_major ? "col" : "row"),
				(b_major == mtk::mateval::col_major ? "col" : "row"),
				(r_major == mtk::mateval::col_major ? "col" : "row"),
				residual,
				(residual < 1e-6 ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m"),
				max_error,
				(max_error < (K * K * 5e-6) ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m")
				);
}

int main() {
	std::printf("----------- Should be passed -----------\n");
	test_AxB(160, 160, 160, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, 200, 200, 200, true);
	test_AxB(160, 160, 160, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, 200, 200, 200, true);
	test_AxB(160, 160, 160, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, 200, 200, 200, true);
	test_AxB(160, 160, 160, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, 200, 200, 200, true);
	test_AxB(160, 160, 160, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, 200, 200, 200, true);
	test_AxB(160, 160, 160, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, 200, 200, 200, true);
	test_AxB(160, 160, 160, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, 200, 200, 200, true);
	test_AxB(160, 160, 160, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, 200, 200, 200, true);
	std::printf("--------- Should not be passed ---------\n");
	test_AxB(160, 160, 160, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, 200, 200, 200, false);
	test_AxB(160, 160, 160, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, 200, 200, 200, false);
	test_AxB(160, 160, 160, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, 200, 200, 200, false);
	test_AxB(160, 160, 160, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, 200, 200, 200, false);
	test_AxB(160, 160, 160, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, 200, 200, 200, false);
	test_AxB(160, 160, 160, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, 200, 200, 200, false);
	test_AxB(160, 160, 160, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, 200, 200, 200, false);
	test_AxB(160, 160, 160, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, 200, 200, 200, false);
}
