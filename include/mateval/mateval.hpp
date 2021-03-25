#ifndef __MATEVAL_MATEVAL_HPP__
#define __MATEVAL_MATEVAL_HPP__

#include <algorithm>
#include <cmath>

namespace mtk {
namespace mateval {

enum major_t {
	col_major = 0,
	row_major = 1,
};

template <class A_T, class B_T, class Func>
void foreach_AxB(
		const unsigned M, const unsigned N, const unsigned K,
		const major_t a_major, const major_t b_major,
		const A_T* const a_ptr, const unsigned lda,
		const B_T* const b_ptr, const unsigned ldb,
		const Func func
		) {
	for (unsigned m = 0; m < M; m++) {
		for (unsigned n = 0; n < M; n++) {
			double c = 0.0;
			for (unsigned k = 0; k < K; k++) {
				// load A
				double a;
				if (a_major == col_major) {
					a = a_ptr[k * lda + m];
				} else {
					a = a_ptr[m * lda + k];
				}

				// load B
				double b;
				if (b_major == col_major) {
					b = b_ptr[k + ldb * n];
				} else {
					b = b_ptr[k * ldb + n];
				}
				c += a * b;
			}
			func(c, m, n);
		}
	}
}

template <class A_T, class B_T, class REF_T>
double residual_AxB(
		const unsigned M, const unsigned N, const unsigned K,
		const major_t a_major, const major_t b_major, const major_t r_major,
		const A_T*   const a_ptr, const unsigned lda,
		const B_T*   const b_ptr, const unsigned ldb,
		const REF_T* const r_ptr, const unsigned ldr
		) {
	double base_norm2 = 0.0;
	double diff_norm2 = 0.0;
	foreach_AxB(
			M, N, K,
			a_major, b_major,
			a_ptr, lda,
			b_ptr, ldb,
			[&](const double c, const unsigned m, const unsigned n) {
				// load Ref
				double r;
				if (r_major == col_major) {
					r = r_ptr[n * ldr + m];
				} else {
					r = r_ptr[m * ldr + n];
				}
				const auto diff = r - c;
				base_norm2 += r * r;
				diff_norm2 += diff * diff;
			});
	return std::sqrt(diff_norm2 / base_norm2);
}

template <class A_T, class B_T, class REF_T>
double max_error_AxB(
		const unsigned M, const unsigned N, const unsigned K,
		const major_t a_major, const major_t b_major, const major_t r_major,
		const A_T*   const a_ptr, const unsigned lda,
		const B_T*   const b_ptr, const unsigned ldb,
		const REF_T* const r_ptr, const unsigned ldr
		) {
	double max_error = 0.0;
	foreach_AxB(
			M, N, K,
			a_major, b_major,
			a_ptr, lda,
			b_ptr, ldb,
			[&](const double c, const unsigned m, const unsigned n) {
				// load Ref
				double r;
				if (r_major == col_major) {
					r = r_ptr[n * ldr + m];
				} else {
					r = r_ptr[m * ldr + n];
				}
				const auto diff = std::abs(r - c);
				max_error = std::max(max_error, diff);
			});
	return max_error;
}

template <class A_T, class REF_T>
double residual(
		const unsigned M, const unsigned N,
		const major_t a_major, const major_t r_major,
		const A_T*   const a_ptr, const unsigned lda,
		const REF_T* const r_ptr, const unsigned ldr
		) {
	double base_norm2 = 0.0;
	double diff_norm2 = 0.0;
	for (unsigned m = 0; m < M; m++) {
		for (unsigned n = 0; n < N; n++) {
			double r, a;
			if (r_major == mtk::mateval::col_major) {
				r = r_ptr[m + n * ldr];
			} else {
				r = r_ptr[n + m * ldr];
			}
			if (a_major == mtk::mateval::col_major) {
				a = a_ptr[m + n * lda];
			} else {
				a = a_ptr[n + m * lda];
			}
			const auto diff = a - r;
			diff_norm2 += diff * diff;
			base_norm2 += r * r;
		}
	}
	return std::sqrt(diff_norm2 / base_norm2);
}

template <class A_T, class REF_T>
double max_error(
		const unsigned M, const unsigned N,
		const major_t a_major, const major_t r_major,
		const A_T*   const a_ptr, const unsigned lda,
		const REF_T* const r_ptr, const unsigned ldr
		) {
	double max_error = 0.0;
	for (unsigned m = 0; m < M; m++) {
		for (unsigned n = 0; n < N; n++) {
			double r, a;
			if (r_major == mtk::mateval::col_major) {
				r = r_ptr[m + n * ldr];
			} else {
				r = r_ptr[n + m * ldr];
			}
			if (a_major == mtk::mateval::col_major) {
				a = a_ptr[m + n * lda];
			} else {
				a = a_ptr[n + m * lda];
			}
			const auto diff = a - r;
			max_error = std::max(max_error, std::abs(diff));
		}
	}
	return max_error;
}

} // namespace mateval
} // namespace mtk
#endif
