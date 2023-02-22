#include <iostream>
#include <memory>
#include <mateval/comparison.hpp>

const std::size_t matrix_dim = 1000;
const std::size_t matrix_ld  = 1200;

using compute_t = double;

void test_AxB(
	const unsigned M,
	const unsigned N,
	const unsigned K,
	const mtk::mateval::layout_t a_major,
	const mtk::mateval::layout_t b_major,
	const mtk::mateval::layout_t r_major,
	const unsigned lda,
	const unsigned ldb,
	const unsigned ldr,
	const bool should_be_passed
	) {
	const std::size_t a_mem_size = lda * (a_major == mtk::mateval::col_major ? K : M);
	const std::size_t b_mem_size = ldb * (b_major == mtk::mateval::col_major ? N : K);
	const std::size_t r_mem_size = ldr * (r_major == mtk::mateval::col_major ? N : M);

	auto mat_a = std::unique_ptr<compute_t[]>(new compute_t [a_mem_size]);
	auto mat_b = std::unique_ptr<compute_t[]>(new compute_t [b_mem_size]);
	auto mat_r = std::unique_ptr<compute_t[]>(new compute_t [r_mem_size]);

	// Set A
	for (unsigned m = 0; m < M; m++) {
		for (unsigned n = 0; n < K; n++) {
			const auto index = (a_major == mtk::mateval::col_major ? (m + n * lda) : (m * lda + n));
			mat_a.get()[index] = (n + 1) * (m + 1) / static_cast<compute_t>(M);
		}
	}
	// Set B
	for (unsigned m = 0; m < K; m++) {
		for (unsigned n = 0; n < N; n++) {
			const auto index = (b_major == mtk::mateval::col_major ? (m + n * ldb) : (m * ldb + n));
			mat_b.get()[index] = (m + 1) * (n + 1) / static_cast<compute_t>(N);
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

	const auto errors = mtk::mateval::get_error_AxB(
		mtk::mateval::max_absolute_error | mtk::mateval::max_relative_error | mtk::mateval::relative_residual | mtk::mateval::avg_relative_error,
		M, N, K,
		a_major, b_major, r_major,
		mat_a.get(), lda,
		mat_b.get(), ldb,
		mat_r.get(), ldr
		);
	const auto residual = errors.at(mtk::mateval::relative_residual);
	const auto max_error = errors.at(mtk::mateval::max_absolute_error);
	const auto max_relative_error = errors.at(mtk::mateval::max_relative_error);
	const auto avg_relative_error = errors.at(mtk::mateval::avg_relative_error);

	std::printf("[%s]{M=%3u,N=%3u,K=%u,lda=%u,ldb=%u,ldr=%u,a_major=%3s,b_major=%3s,r_major=%3s} residual=%e(%6s), max_error=%e(%6s), max_relative_error=%e(%6s), avg_relative_error=%e(%6s)\n",
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
				((avg_relative_error < 1e-6 == should_be_passed) ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m"),
				avg_relative_error,
				((max_relative_error < 1e-6 == should_be_passed) ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m")
				);
}

void test_A(
	const unsigned M,
	const unsigned N,
	const mtk::mateval::layout_t a_major,
	const mtk::mateval::layout_t r_major,
	const unsigned lda,
	const unsigned ldr,
	const bool should_be_passed
	) {
	const std::size_t a_mem_size = (a_major == mtk::mateval::col_major ? N : M) * lda;
	const std::size_t r_mem_size = (r_major == mtk::mateval::col_major ? N : M) * ldr;

	auto mat_a = std::unique_ptr<compute_t[]>(new compute_t [a_mem_size]);
	auto mat_r = std::unique_ptr<double[]>(new double [r_mem_size]);

	// Set R and A
	for (unsigned m = 0; m < M; m++) {
		for (unsigned n = 0; n < N; n++) {
			const auto index_r = (r_major == mtk::mateval::col_major ? (m + n * ldr) : (m * ldr + n));
			const auto index_a = (a_major == mtk::mateval::col_major ? (m + n * lda) : (m * lda + n));
			mat_r.get()[index_r] = (n + 1) * (m + 1) / static_cast<compute_t>(M);
			mat_a.get()[index_a] = static_cast<compute_t>(mat_r.get()[index_r] * (should_be_passed ? 1.0f : -1.0f));
		}
	}

	const auto errors = mtk::mateval::get_error(
		mtk::mateval::max_absolute_error | mtk::mateval::max_relative_error | mtk::mateval::relative_residual | mtk::mateval::avg_relative_error,
		M, N,
		a_major, r_major,
		mat_a.get(), lda,
		mat_r.get(), ldr
		);
	const auto residual = errors.at(mtk::mateval::relative_residual);
	const auto max_error = errors.at(mtk::mateval::max_absolute_error);
	const auto max_relative_error = errors.at(mtk::mateval::max_relative_error);
	const auto avg_relative_error = errors.at(mtk::mateval::avg_relative_error);

	std::printf("[%s]{M=%3u,N=%3u,lda=%u,ldr=%u,a_major=%3s,r_major=%3s} residual=%e(%6s), max_error=%e(%6s), max_relative_error=%e(%6s), avg_relative_error=%e(%6s)\n",
				__func__,
				M, N,
				lda, ldr,
				(a_major == mtk::mateval::col_major ? "col" : "row"),
				(r_major == mtk::mateval::col_major ? "col" : "row"),
				residual,
				((residual < 1e-6) == should_be_passed ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m"),
				max_error,
				((max_error < 5e-6) == should_be_passed ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m"),
				max_relative_error,
				((avg_relative_error < 5e-6) == should_be_passed ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m"),
				avg_relative_error,
				((max_relative_error < 5e-6) == should_be_passed ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m")
				);
}

void test_UxSxVt(
	const unsigned M,
	const unsigned N,
	const unsigned K,
	const mtk::mateval::layout_t u_major,
	const mtk::mateval::layout_t v_major,
	const mtk::mateval::layout_t r_major,
	const unsigned ldu,
	const unsigned ldv,
	const unsigned ldr,
	const bool should_be_passed
	) {
	const std::size_t u_mem_size = ldu * (u_major == mtk::mateval::col_major ? K : M);
	const std::size_t s_mem_size = K;
	const std::size_t v_mem_size = ldv * (v_major == mtk::mateval::col_major ? K : N);
	const std::size_t r_mem_size = ldr * (r_major == mtk::mateval::col_major ? N : M);

	auto mat_u = std::unique_ptr<compute_t[]>(new compute_t [u_mem_size]);
	auto mat_s = std::unique_ptr<compute_t[]>(new compute_t [s_mem_size]);
	auto mat_v = std::unique_ptr<compute_t[]>(new compute_t [v_mem_size]);
	auto mat_r = std::unique_ptr<compute_t[]>(new compute_t [r_mem_size]);

	// Set U
	for (unsigned m = 0; m < M; m++) {
		for (unsigned n = 0; n < K; n++) {
			const auto index = (u_major == mtk::mateval::col_major ? (m + n * ldu) : (m * ldu + n));
			mat_u.get()[index] = (n + 1) * (m + 1) / static_cast<compute_t>(M);
		}
	}
	// Set S
	for (unsigned m = 0; m < K; m++) {
		mat_s.get()[m] = (m + 1) / static_cast<double>(K);
	}
	// Set V
	for (unsigned m = 0; m < K; m++) {
		for (unsigned n = 0; n < N; n++) {
			const auto index = (v_major == mtk::mateval::col_major ? (m * ldv + n) : (m + n * ldv));
			mat_v.get()[index] = (m + 1) * (n + 1) / static_cast<compute_t>(N);
		}
	}
	// Set ref
	for (unsigned m = 0; m < M; m++) {
		for (unsigned n = 0; n < N; n++) {
			const auto index = (r_major == mtk::mateval::col_major ? (m + n * ldr) : (m * ldr + n));
			mat_r.get()[index] = static_cast<double>(K * K) * (K + 1) * (K + 1) / 4 * (m + 1) * (n + 1) / (static_cast<double>(M) * K * N);
			if (!should_be_passed) {
				mat_r.get()[index] *= -1.0f;
			}
		}
	}

	const auto residual = mtk::mateval::residual_UxSxVt(
		M, N, K,
		u_major, v_major, r_major,
		mat_u.get(), ldu,
		mat_s.get(),
		mat_v.get(), ldv,
		mat_r.get(), ldr
		);

	std::printf("[%s]{M=%3u,N=%3u,K=%u,ldu=%u,ldv=%u,ldr=%u,u_major=%3s,v_major=%3s,r_major=%3s} residual=%e(%6s)\n",
				__func__,
				M, N, K,
				ldu, ldv, ldr,
				(u_major == mtk::mateval::col_major ? "col" : "row"),
				(v_major == mtk::mateval::col_major ? "col" : "row"),
				(r_major == mtk::mateval::col_major ? "col" : "row"),
				residual,
				((residual < 1e-6) == should_be_passed ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m")
				);
}

void test_orthogonality(
	const unsigned M,
	const unsigned N,
	const mtk::mateval::layout_t major,
	const unsigned ld,
	const bool should_be_passed
	) {
	const std::size_t r_mem_size = ld * (major == mtk::mateval::col_major ? N : M);
	const std::size_t vec_length = std::max(M, N);

	auto mat_r = std::unique_ptr<compute_t[]>(new compute_t [r_mem_size]);
	auto vec_r = std::unique_ptr<compute_t[]>(new compute_t [vec_length]);

	double vec_norm2 = 0.;
	for (unsigned i = 0; i < vec_length; i++) {
		vec_r.get()[i] = i / static_cast<double>(vec_length);
		vec_norm2 += vec_r.get()[i] * vec_r.get()[i];
	}

	// Set H
	for (unsigned m = 0; m < M; m++) {
		for (unsigned n = 0; n < N; n++) {
			const auto index = (major == mtk::mateval::col_major ? (m + n * ld) : (m * ld + n));
			const auto v = (m == n ? 1.0 : 0.0) - 2.0 * vec_r.get()[m] * vec_r.get()[n] / vec_norm2;
			mat_r.get()[index] = v + (should_be_passed ? 0. : 1.);
		}
	}

	const auto orthogonality = mtk::mateval::orthogonality(
			M, N,
			major,
			mat_r.get(), ld
			);

	std::printf("[%s]{M=%3u,N=%3u,ldr=%u,major=%3s} orthogonality=%e(%6s)\n",
				__func__,
				M, N,
				ld,
				(major == mtk::mateval::col_major ? "col" : "row"),
				orthogonality,
				((orthogonality < 1e-6) == should_be_passed ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m")
				);
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
	test_A(matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, true);
	test_A(matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, true);
	test_A(matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, true);
	test_A(matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, true);
	test_A(matrix_dim / 2, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, true);
	test_A(matrix_dim / 2, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, true);
	test_A(matrix_dim / 2, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, true);
	test_A(matrix_dim / 2, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, true);
	test_A(matrix_dim, matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, true);
	test_A(matrix_dim, matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, true);
	test_A(matrix_dim, matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, true);
	test_A(matrix_dim, matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, true);
	test_orthogonality(matrix_dim, matrix_dim, mtk::mateval::col_major, matrix_dim, true);
	test_orthogonality(matrix_dim, matrix_dim, mtk::mateval::row_major, matrix_dim, true);
	test_orthogonality(matrix_dim, matrix_dim / 2, mtk::mateval::col_major, matrix_dim, true);
	test_orthogonality(matrix_dim, matrix_dim / 2, mtk::mateval::row_major, matrix_dim, true);
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
	test_A(matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, false);
	test_A(matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, false);
	test_A(matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, false);
	test_A(matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, false);
	test_A(matrix_dim / 2, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, false);
	test_A(matrix_dim / 2, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, false);
	test_A(matrix_dim / 2, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, false);
	test_A(matrix_dim / 2, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, false);
	test_A(matrix_dim, matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, false);
	test_A(matrix_dim, matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, false);
	test_A(matrix_dim, matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, false);
	test_A(matrix_dim, matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim / 2, matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim, matrix_dim / 2, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_UxSxVt(matrix_dim, matrix_dim, matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, false);
	test_orthogonality(matrix_dim, matrix_dim, mtk::mateval::col_major, matrix_dim, false);
	test_orthogonality(matrix_dim, matrix_dim, mtk::mateval::row_major, matrix_dim, false);
	test_orthogonality(matrix_dim, matrix_dim / 2, mtk::mateval::col_major, matrix_dim, false);
	test_orthogonality(matrix_dim, matrix_dim / 2, mtk::mateval::row_major, matrix_dim, false);
}
