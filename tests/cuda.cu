#include <iostream>
#include <type_traits>
#include <complex>
#include <memory>
#include <cublas.h>
#include <mateval/cuda/comparison.hpp>

const std::size_t matrix_dim = 1000;
const std::size_t matrix_ld  = 1200;
using compute_t = double;

namespace {
inline cuDoubleComplex operator*(const cuDoubleComplex a, const cuDoubleComplex b) {
	return make_cuDoubleComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
inline cuDoubleComplex operator+(const cuDoubleComplex a, const cuDoubleComplex b) {
	return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
}
inline cuDoubleComplex operator*=(cuDoubleComplex& a, const double b) {
	return a = make_cuDoubleComplex(a.x * b, a.y * b);
}

template <class T>
std::string dtype_str();
template <>
std::string dtype_str<double>() {return "double";}
template <>
std::string dtype_str<cuDoubleComplex>() {return "cuDoubleComplex";}
} // unnamed namespace

template <class compute_t>
int test_GEMM(
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

	compute_t alpha, beta;

	if constexpr (std::is_same_v<compute_t, double>) {
		alpha = -0.5;
		beta = 0.5;

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
	} else if constexpr (std::is_same_v<compute_t, cuDoubleComplex>) {
		alpha = make_cuDoubleComplex(-0.25, 0.25);
		beta = make_cuDoubleComplex(0.25, -0.25);

		using real_t = double;

		// Set A
		for (unsigned m = 0; m < M; m++) {
			for (unsigned n = 0; n < K; n++) {
				const auto index = (a_major == mtk::mateval::col_major ? (m + n * lda) : (m * lda + n));
				const auto v = (n + 1) * (m + 1) / static_cast<real_t>(M);
				mat_a.get()[index] = make_cuDoubleComplex(v, 2 * v);
			}
		}
		// Set B
		for (unsigned m = 0; m < K; m++) {
			for (unsigned n = 0; n < N; n++) {
				const auto index = (b_major == mtk::mateval::col_major ? (m + n * ldb) : (m * ldb + n));
				const auto v = (n + 1) * (m + 1) / static_cast<real_t>(N);
				mat_b.get()[index] = make_cuDoubleComplex(v, 4 * v);
			}
		}
		// Set C
		for (unsigned m = 0; m < M; m++) {
			for (unsigned n = 0; n < N; n++) {
				const auto index = (c_major == mtk::mateval::col_major ? (m + n * ldc) : (m * ldc + n));
				const auto v = (n + 1) * (m + 1) / static_cast<real_t>(N);
				mat_c.get()[index] = make_cuDoubleComplex(2 * v, v);
			}
		}
		// Set ref
		for (unsigned m = 0; m < M; m++) {
			for (unsigned n = 0; n < N; n++) {
				const auto r_index = (r_major == mtk::mateval::col_major ? (m + n * ldr) : (m * ldr + n));
				const auto v = static_cast<real_t>(K * (K + 1) * (2 * K + 1) / 6) * (m + 1) * (n + 1) / static_cast<real_t>(M * N);
				const auto w = make_cuDoubleComplex(-7 * v, 6 * v);

				const auto c_index = (c_major == mtk::mateval::col_major ? (m + n * ldc) : (m * ldc + n));
				mat_r.get()[r_index] = alpha * w + beta * mat_c.get()[c_index];

				if (!should_be_passed) {
					mat_r.get()[r_index] *= -1.0;
				}
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

	const auto check_relative_residual = (residual < 1e-13) == should_be_passed;
	const auto check_max_error = max_error < (K * K * K * 5e-10) == should_be_passed;
	const auto check_max_relative_error = max_relative_error < (1e-13) == should_be_passed;
	const auto check_avg_relative_error = avg_relative_error < (1e-13) == should_be_passed;

	std::printf("[%s]{dtype=%s,M=%3u,N=%3u,K=%u,lda=%u,ldb=%u,ldr=%u,a_major=%3s,b_major=%3s,c_major=%3s,r_major=%3s} residual=%e(%6s), max_error=%e(%6s), max_relative_error=%e(%6s), avg_relative_error=%e(%6s)\n",
				__func__,
				dtype_str<compute_t>().c_str(),
				M, N, K,
				lda, ldb, ldr,
				(a_major == mtk::mateval::col_major ? "col" : "row"),
				(b_major == mtk::mateval::col_major ? "col" : "row"),
				(c_major == mtk::mateval::col_major ? "col" : "row"),
				(r_major == mtk::mateval::col_major ? "col" : "row"),
				residual,
				check_relative_residual ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m",
				max_error,
				check_max_error ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m",
				max_relative_error,
				check_max_relative_error ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m",
				avg_relative_error,
				check_avg_relative_error ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m"
				);

	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
	cudaFree(dr);

	return !(check_relative_residual && check_max_error && check_max_relative_error && check_avg_relative_error);
}

int test_A_B(
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

	const auto check_relative_residual = (residual < 1e-13) == should_be_passed;
	const auto check_max_error = max_error < (M * N * 5e-10) == should_be_passed;
	const auto check_max_relative_error = max_relative_error < (1e-13) == should_be_passed;
	const auto check_avg_relative_error = avg_relative_error < (1e-13) == should_be_passed;

	std::printf("[%s]{M=%3u,N=%3u,lda=%u,ldb=%u,a_major=%3s,b_major=%3s} residual=%e(%6s), max_error=%e(%6s), max_relative_error=%e(%6s), avg_relative_error=%e(%6s)\n",
				__func__,
				M, N,
				lda, ldb,
				(a_major == mtk::mateval::col_major ? "col" : "row"),
				(b_major == mtk::mateval::col_major ? "col" : "row"),
				residual,
				check_relative_residual ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m",
				max_error,
				check_max_error ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m",
				max_relative_error,
				check_max_relative_error ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m",
				avg_relative_error,
				check_avg_relative_error ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m"
				);

	cudaFree(da);
	cudaFree(db);

	return !(check_relative_residual && check_max_error && check_max_relative_error && check_avg_relative_error);
}

int test_UxSxVt(
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

	const auto check_relative_error = (residual < 1e-6) == should_be_passed;

	std::printf("[%s]{M=%3u,N=%3u,K=%u,ldu=%u,ldv=%u,ldr=%u,u_major=%3s,v_major=%3s,r_major=%3s} residual=%e(%6s)\n",
				__func__,
				M, N, K,
				ldu, ldv, ldr,
				(u_major == mtk::mateval::col_major ? "col" : "row"),
				(v_major == mtk::mateval::col_major ? "col" : "row"),
				(r_major == mtk::mateval::col_major ? "col" : "row"),
				residual,
				(check_relative_error ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m")
				);

	cudaFree(du);
	cudaFree(ds);
	cudaFree(dv);
	cudaFree(dr);

	return !check_relative_error;
}

int test_orthogonality(
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

	const auto check_orth = (orthogonality < 1e-6) == should_be_passed;

	std::printf("[%s]{M=%3u,N=%3u,ldr=%u,major=%3s} orthogonality=%e(%6s)\n",
				__func__,
				M, N,
				ld,
				(major == mtk::mateval::col_major ? "col" : "row"),
				orthogonality,
				(check_orth ? "\x1B[32mPASSED\x1B[37m" : "\x1B[31mFAILED\x1B[37m")
				);

	cudaFree(dr);

	return !check_orth;
}

std::pair<std::uint32_t, std::uint32_t>
test(bool should_be_passed) {
	std::uint32_t num_tests = 0;
	std::uint32_t num_failed = 0;
	// GEMM test
	{
		std::vector<std::vector<std::uint32_t>> gemm_shape_list = {{matrix_dim, matrix_dim, matrix_dim}, {matrix_dim / 2, matrix_dim, matrix_dim}, {matrix_dim, matrix_dim / 2, matrix_dim}, {matrix_dim, matrix_dim, matrix_dim / 2}};
		std::vector<mtk::mateval::layout_t> gemm_layout_list_real = {mtk::mateval::col_major, mtk::mateval::row_major};
		for (const auto &shape : gemm_shape_list) {
			for (const auto op_A : gemm_layout_list_real) {
				for (const auto op_B : gemm_layout_list_real) {
					for (const auto op_C : gemm_layout_list_real) {
						for (const auto op_R : gemm_layout_list_real) {
							if (test_GEMM<double>(shape[0], shape[1], shape[2] , op_A, op_B, op_C, op_R, matrix_ld, matrix_ld, matrix_ld, matrix_ld, should_be_passed)) {
								num_failed++;
							}
							num_tests++;
						}
					}
				}
			}
		}
		std::vector<mtk::mateval::layout_t> gemm_layout_list_complex = {mtk::mateval::col_major, mtk::mateval::row_major /*, mtk::mateval::conj*/};
		for (const auto &shape : gemm_shape_list) {
			for (const auto op_A : gemm_layout_list_complex) {
				for (const auto op_B : gemm_layout_list_complex) {
					for (const auto op_C : gemm_layout_list_complex) {
						for (const auto op_R : gemm_layout_list_real) {
							if (test_GEMM<cuDoubleComplex>(shape[0], shape[1], shape[2] , op_A, op_B, op_C, op_R, matrix_ld, matrix_ld, matrix_ld, matrix_ld, should_be_passed)) {
								num_failed++;
							}
						}
						num_tests++;
					}
				}
			}
		}
	}
	// Simple comparison test
	{
		std::vector<std::vector<std::uint32_t>> matrix_shape_list = {{matrix_dim, matrix_dim}, {matrix_dim / 2, matrix_dim}, {matrix_dim, matrix_dim / 2}};
		std::vector<mtk::mateval::layout_t> matrix_layout_list_real = {mtk::mateval::col_major, mtk::mateval::row_major};
		for (const auto &shape : matrix_shape_list) {
			for (const auto op_A : matrix_layout_list_real) {
				for (const auto op_B : matrix_layout_list_real) {
					if (test_A_B(shape[0], shape[1], op_A, op_B, matrix_ld, matrix_ld, should_be_passed)) {
						num_failed++;
					}
				}
				num_tests++;
			}
		}
	}
	// SVD test
	{
		std::vector<std::vector<std::uint32_t>> decomp_shape_list = {{matrix_dim, matrix_dim, matrix_dim}, {matrix_dim / 2, matrix_dim, matrix_dim}, {matrix_dim, matrix_dim / 2, matrix_dim}, {matrix_dim, matrix_dim, matrix_dim / 2}};
		std::vector<mtk::mateval::layout_t> matrix_layout_list_real = {mtk::mateval::col_major, mtk::mateval::row_major};
		for (const auto &shape : decomp_shape_list) {
			for (const auto op_U : matrix_layout_list_real) {
				for (const auto op_V : matrix_layout_list_real) {
					for (const auto op_A : matrix_layout_list_real) {
						if (test_UxSxVt(shape[0], shape[1], shape[2], op_U, op_V, op_A, matrix_ld, matrix_ld, matrix_ld, should_be_passed)) {
							num_failed++;
						}
					}
					num_tests++;
				}
			}
		}
	}
	// Orthogonality test
	{
		std::vector<std::vector<std::uint32_t>> matrix_shape_list = {{matrix_dim, matrix_dim}, {matrix_dim, matrix_dim / 2}};
		std::vector<mtk::mateval::layout_t> matrix_layout_list_real = {mtk::mateval::col_major, mtk::mateval::row_major};
		for (const auto &shape : matrix_shape_list) {
			for (const auto op_Q : matrix_layout_list_real) {
				if (test_orthogonality(shape[0], shape[1], op_Q, matrix_ld, should_be_passed)) {
					num_failed++;
				}
			}
			num_tests++;
		}
	}

	return std::make_pair(num_failed, num_tests);
}

int main() {
	std::printf("----------- passing test -----------\n");
	const auto [F0, T0] = test(true);
	std::printf("--------- failing test ---------\n");
	const auto [F1, T1] = test(false);

	const auto num_tests = T0 + T1;
	const auto num_failed = F0 + F1;

	std::printf("[RESULT] %5u / %5u passed\n", num_tests - num_failed, num_tests);

	if (num_failed) {
		return 1;
	}
	return 0;
}
