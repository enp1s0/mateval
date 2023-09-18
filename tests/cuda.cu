#include <iostream>
#include <memory>
#include <cublas.h>
#include <mateval/cuda/comparison.hpp>

const std::size_t matrix_dim = 1000;
const std::size_t matrix_ld  = 1200;
using compute_t = double;

void test_GEMM(
	const unsigned M,
	const unsigned N,
	const unsigned K,
	const mtk::mateval::layout_t a_major,
	const mtk::mateval::layout_t b_major,
	const mtk::mateval::layout_t c_major,
	const mtk::mateval::layout_t r_major,
	const unsigned lda,
	const unsigned ldb,
	const unsigned ldc,
	const unsigned ldr,
	const bool should_be_passed
	) {
	const std::size_t a_mem_size = lda * (a_major == mtk::mateval::col_major ? K : M);
	const std::size_t b_mem_size = ldb * (b_major == mtk::mateval::col_major ? N : K);
	const std::size_t c_mem_size = ldc * (c_major == mtk::mateval::col_major ? N : M);
	const std::size_t r_mem_size = ldr * (r_major == mtk::mateval::col_major ? N : M);

	auto mat_a = std::unique_ptr<compute_t[]>(new compute_t [a_mem_size]);
	auto mat_b = std::unique_ptr<compute_t[]>(new compute_t [b_mem_size]);
	auto mat_c = std::unique_ptr<compute_t[]>(new compute_t [c_mem_size]);
	auto mat_r = std::unique_ptr<compute_t[]>(new compute_t [r_mem_size]);

	compute_t *da, *db, *dc, *dr;
	cudaMalloc(&da, sizeof(compute_t) * a_mem_size);
	cudaMalloc(&db, sizeof(compute_t) * b_mem_size);
	cudaMalloc(&dc, sizeof(compute_t) * c_mem_size);
	cudaMalloc(&dr, sizeof(compute_t) * r_mem_size);

	const compute_t alpha = -0.5, beta = 0.5;

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
	// Set C
	for (unsigned m = 0; m < M; m++) {
		for (unsigned n = 0; n < N; n++) {
			const auto index = (c_major == mtk::mateval::col_major ? (m + n * ldc) : (m * ldc + n));
			mat_c.get()[index] = (m + 1) * (n + 1) / static_cast<compute_t>(N);
		}
	}
	// Set ref
	for (unsigned m = 0; m < M; m++) {
		for (unsigned n = 0; n < N; n++) {
			const auto r_index = (r_major == mtk::mateval::col_major ? (m + n * ldr) : (m * ldr + n));
			mat_r.get()[r_index] = static_cast<double>(K * (K + 1) * (2 * K + 1) / 6) * (m + 1) * (n + 1) / static_cast<double>(M * N);

			const auto c_index = (c_major == mtk::mateval::col_major ? (m + n * ldc) : (m * ldc + n));
			mat_r.get()[r_index] = alpha * mat_r.get()[r_index] + beta * mat_c.get()[c_index];

			if (!should_be_passed) {
				mat_r.get()[r_index] *= -1.0f;
			}
		}
	}

	cudaMemcpy(da, mat_a.get(), sizeof(compute_t) * a_mem_size, cudaMemcpyDefault);
	cudaMemcpy(db, mat_b.get(), sizeof(compute_t) * b_mem_size, cudaMemcpyDefault);
	cudaMemcpy(dc, mat_c.get(), sizeof(compute_t) * c_mem_size, cudaMemcpyDefault);
	cudaMemcpy(dr, mat_r.get(), sizeof(compute_t) * r_mem_size, cudaMemcpyDefault);

	const auto errors = mtk::mateval::cuda::get_error_GEMM(
		mtk::mateval::max_absolute_error | mtk::mateval::max_relative_error | mtk::mateval::relative_residual | mtk::mateval::avg_relative_error,
		M, N, K,
		a_major, b_major, c_major, r_major,
		alpha,
		da, lda,
		db, ldb,
		beta,
		dc, ldc,
		dr, ldr
		);
	const auto residual = errors.at(mtk::mateval::relative_residual);
	const auto max_error = errors.at(mtk::mateval::max_absolute_error);
	const auto max_relative_error = errors.at(mtk::mateval::max_relative_error);
	const auto avg_relative_error = errors.at(mtk::mateval::avg_relative_error);
	std::printf("[%s]{M=%3u,N=%3u,K=%u,lda=%u,ldb=%u,ldr=%u,a_major=%3s,b_major=%3s,c_major=%3s,r_major=%3s,alpha=%e,beta=%e} residual=%e(%6s), max_error=%e(%6s), max_relative_error=%e(%6s), avg_relative_error=%e(%6s)\n",
				__func__,
				M, N, K,
				lda, ldb, ldr,
				(a_major == mtk::mateval::col_major ? "col" : "row"),
				(b_major == mtk::mateval::col_major ? "col" : "row"),
				(c_major == mtk::mateval::col_major ? "col" : "row"),
				(r_major == mtk::mateval::col_major ? "col" : "row"),
				alpha, beta,
				residual,
				((residual < 1e-6) == should_be_passed ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m"),
				max_error,
				((max_error < (K * K * K * 5e-8) == should_be_passed) ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m"),
				max_relative_error,
				((max_relative_error < (1e-6) == should_be_passed) ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m"),
				avg_relative_error,
				((avg_relative_error < (1e-6) == should_be_passed) ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m")
				);

	cudaFree(da);
	cudaFree(db);
}

void test_A_B(
	const unsigned M,
	const unsigned N,
	const mtk::mateval::layout_t a_major,
	const mtk::mateval::layout_t b_major,
	const unsigned lda,
	const unsigned ldb,
	const bool should_be_passed
	) {
	const std::size_t a_mem_size = lda * (a_major == mtk::mateval::col_major ? N : M);
	const std::size_t b_mem_size = ldb * (b_major == mtk::mateval::col_major ? N : M);

	auto mat_a = std::unique_ptr<compute_t[]>(new compute_t [a_mem_size]);
	auto mat_b = std::unique_ptr<compute_t[]>(new compute_t [b_mem_size]);

	compute_t *da, *db;
	cudaMalloc(&da, sizeof(compute_t) * a_mem_size);
	cudaMalloc(&db, sizeof(compute_t) * b_mem_size);

	// Set A
	for (unsigned m = 0; m < M; m++) {
		for (unsigned n = 0; n < N; n++) {
			const auto index = (a_major == mtk::mateval::col_major ? (m + n * lda) : (m * lda + n));
			mat_a.get()[index] = (n + 1) * (m + 1) / static_cast<compute_t>(M);
		}
	}
	// Set B
	for (unsigned m = 0; m < M; m++) {
		for (unsigned n = 0; n < N; n++) {
			const auto index = (b_major == mtk::mateval::col_major ? (m + n * ldb) : (m * ldb + n));
			mat_b.get()[index] = (m + 1) * (n + 1) / static_cast<compute_t>(M) + (should_be_passed ? 0 : 1);
		}
	}

	cudaMemcpy(da, mat_a.get(), sizeof(compute_t) * a_mem_size, cudaMemcpyDefault);
	cudaMemcpy(db, mat_b.get(), sizeof(compute_t) * b_mem_size, cudaMemcpyDefault);

	const auto errors = mtk::mateval::cuda::get_error(
		mtk::mateval::max_absolute_error | mtk::mateval::max_relative_error | mtk::mateval::relative_residual | mtk::mateval::avg_relative_error,
		M, N,
		a_major, b_major,
		da, lda,
		db, ldb
		);
	const auto residual = errors.at(mtk::mateval::relative_residual);
	const auto max_error = errors.at(mtk::mateval::max_absolute_error);
	const auto max_relative_error = errors.at(mtk::mateval::max_relative_error);
	const auto avg_relative_error = errors.at(mtk::mateval::avg_relative_error);
	std::printf("[%s]{M=%3u,N=%3u,lda=%u,ldb=%u,a_major=%3s,b_major=%3s} residual=%e(%6s), max_error=%e(%6s), max_relative_error=%e(%6s), avg_relative_error=%e(%6s)\n",
				__func__,
				M, N,
				lda, ldb,
				(a_major == mtk::mateval::col_major ? "col" : "row"),
				(b_major == mtk::mateval::col_major ? "col" : "row"),
				residual,
				((residual < 1e-6) == should_be_passed ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m"),
				max_error,
				((max_error < (M * M * 5e-8) == should_be_passed) ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m"),
				max_relative_error,
				((max_relative_error < (1e-6) == should_be_passed) ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m"),
				avg_relative_error,
				((avg_relative_error < (1e-6) == should_be_passed) ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m")
				);

	cudaFree(da);
	cudaFree(db);
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

	compute_t *du, *ds, *dv, *dr;
	cudaMalloc(&du, sizeof(compute_t) * u_mem_size);
	cudaMalloc(&ds, sizeof(compute_t) * s_mem_size);
	cudaMalloc(&dv, sizeof(compute_t) * v_mem_size);
	cudaMalloc(&dr, sizeof(compute_t) * r_mem_size);

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

	cudaMemcpy(du, mat_u.get(), sizeof(compute_t) * u_mem_size, cudaMemcpyDefault);
	cudaMemcpy(ds, mat_s.get(), sizeof(compute_t) * s_mem_size, cudaMemcpyDefault);
	cudaMemcpy(dv, mat_v.get(), sizeof(compute_t) * v_mem_size, cudaMemcpyDefault);
	cudaMemcpy(dr, mat_r.get(), sizeof(compute_t) * r_mem_size, cudaMemcpyDefault);

	const auto residual = mtk::mateval::cuda::residual_UxSxVt(
		M, N, K,
		u_major, v_major, r_major,
		du, ldu,
		ds,
		dv, ldv,
		dr, ldr
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

	cudaFree(du);
	cudaFree(ds);
	cudaFree(dv);
	cudaFree(dr);
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

	compute_t *dr;
	cudaMalloc(&dr, sizeof(compute_t) * r_mem_size);

	// Set H
	for (unsigned m = 0; m < M; m++) {
		for (unsigned n = 0; n < N; n++) {
			const auto index = (major == mtk::mateval::col_major ? (m + n * ld) : (m * ld + n));
			const auto v = (m == n ? 1.0 : 0.0) - 2.0 * vec_r.get()[m] * vec_r.get()[n] / vec_norm2;
			mat_r.get()[index] = v + (should_be_passed ? 0. : 1.);
		}
	}

	cudaMemcpy(dr, mat_r.get(), sizeof(compute_t) * r_mem_size, cudaMemcpyDefault);

	const auto orthogonality = mtk::mateval::cuda::orthogonality(
			M, N,
			major,
			dr, ld
			);

	std::printf("[%s]{M=%3u,N=%3u,ldr=%u,major=%3s} orthogonality=%e(%6s)\n",
				__func__,
				M, N,
				ld,
				(major == mtk::mateval::col_major ? "col" : "row"),
				orthogonality,
				((orthogonality < 1e-6) == should_be_passed ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m")
				);

	cudaFree(dr);
}

int main() {
	std::printf("----------- passing test -----------\n");
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, true);
	test_A_B(matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, true);
	test_A_B(matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, true);
	test_A_B(matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, true);
	test_A_B(matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, true);
	test_A_B(matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, true);
	test_A_B(matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, true);
	test_A_B(matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, true);
	test_A_B(matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, true);
	test_A_B(matrix_dim / 2, matrix_dim    , mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, true);
	test_A_B(matrix_dim / 2, matrix_dim    , mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, true);
	test_A_B(matrix_dim / 2, matrix_dim    , mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, true);
	test_A_B(matrix_dim / 2, matrix_dim    , mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, true);
	test_A_B(matrix_dim / 2, matrix_dim    , mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, true);
	test_A_B(matrix_dim / 2, matrix_dim    , mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, true);
	test_A_B(matrix_dim / 2, matrix_dim    , mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, true);
	test_A_B(matrix_dim / 2, matrix_dim    , mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, true);
	test_A_B(matrix_dim    , matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, true);
	test_A_B(matrix_dim    , matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, true);
	test_A_B(matrix_dim    , matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, true);
	test_A_B(matrix_dim    , matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, true);
	test_A_B(matrix_dim    , matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, true);
	test_A_B(matrix_dim    , matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, true);
	test_A_B(matrix_dim    , matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, true);
	test_A_B(matrix_dim    , matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, true);
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
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim / 2, matrix_dim    , matrix_dim    , mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim / 2, matrix_dim    , mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_GEMM(matrix_dim    , matrix_dim    , matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, matrix_ld, matrix_ld, false);
	test_A_B(matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, false);
	test_A_B(matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, false);
	test_A_B(matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, false);
	test_A_B(matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, false);
	test_A_B(matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, false);
	test_A_B(matrix_dim, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, false);
	test_A_B(matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, false);
	test_A_B(matrix_dim, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, false);
	test_A_B(matrix_dim / 2, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, false);
	test_A_B(matrix_dim / 2, matrix_dim, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, false);
	test_A_B(matrix_dim / 2, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, false);
	test_A_B(matrix_dim / 2, matrix_dim, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, false);
	test_A_B(matrix_dim / 2, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, false);
	test_A_B(matrix_dim / 2, matrix_dim, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, false);
	test_A_B(matrix_dim / 2, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, false);
	test_A_B(matrix_dim / 2, matrix_dim, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, false);
	test_A_B(matrix_dim, matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, false);
	test_A_B(matrix_dim, matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::col_major, matrix_ld, matrix_ld, false);
	test_A_B(matrix_dim, matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, false);
	test_A_B(matrix_dim, matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::col_major, matrix_ld, matrix_ld, false);
	test_A_B(matrix_dim, matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, false);
	test_A_B(matrix_dim, matrix_dim / 2, mtk::mateval::col_major, mtk::mateval::row_major, matrix_ld, matrix_ld, false);
	test_A_B(matrix_dim, matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, false);
	test_A_B(matrix_dim, matrix_dim / 2, mtk::mateval::row_major, mtk::mateval::row_major, matrix_ld, matrix_ld, false);
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
