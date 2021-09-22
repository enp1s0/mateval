#ifndef __MATEVAL_COMPARISON_HPP__
#define __MATEVAL_COMPARISON_HPP__

#include <algorithm>
#include <cmath>
#include <utility>
#include <tuple>
#include "common.hpp"

namespace mtk {
namespace mateval {

template <class A_T, class B_T, class Func>
void foreach_AxB(
		const unsigned M, const unsigned N, const unsigned K,
		const major_t a_major, const major_t b_major,
		const A_T* const a_ptr, const unsigned lda,
		const B_T* const b_ptr, const unsigned ldb,
		const Func func
		) {
#pragma omp parallel for collapse(2)
	for (unsigned m = 0; m < M; m++) {
		for (unsigned n = 0; n < N; n++) {
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
#pragma omp critical
			{func(c, m, n);}
		}
	}
}

template <class A_T, class B_T, class Func>
void foreach_AxB_with_abs(
		const unsigned M, const unsigned N, const unsigned K,
		const major_t a_major, const major_t b_major,
		const A_T* const a_ptr, const unsigned lda,
		const B_T* const b_ptr, const unsigned ldb,
		const Func func
		) {
#pragma omp parallel for collapse(2)
	for (unsigned m = 0; m < M; m++) {
		for (unsigned n = 0; n < N; n++) {
			double c = 0.0;
			double abs_c = 0.0;
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
				abs_c += std::abs(a) * std::abs(b);
			}
#pragma omp critical
			{func(c, abs_c, m, n);}
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
				base_norm2 += c * c;
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

template <class A_T, class B_T, class REF_T>
double max_relative_error_AxB(
		const unsigned M, const unsigned N, const unsigned K,
		const major_t a_major, const major_t b_major, const major_t r_major,
		const A_T*   const a_ptr, const unsigned lda,
		const B_T*   const b_ptr, const unsigned ldb,
		const REF_T* const r_ptr, const unsigned ldr
		) {
	double max_error = 0.0;
	double max_element = 0.0;
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
				max_error = std::max(std::abs(r - c), max_error);
				max_element = std::max(std::abs(c), max_element);
			});
	return max_error / max_element;
}

template <class A_T, class B_T, class REF_T>
std::tuple<double, double> max_error_and_residual_AxB(
		const unsigned M, const unsigned N, const unsigned K,
		const major_t a_major, const major_t b_major, const major_t r_major,
		const A_T*   const a_ptr, const unsigned lda,
		const B_T*   const b_ptr, const unsigned ldb,
		const REF_T* const r_ptr, const unsigned ldr
		) {
	double max_error = 0.0;
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
				const auto diff = std::abs(r - c);
				max_error = std::max(max_error, diff);
				base_norm2 += c * c;
				diff_norm2 += diff * diff;
			});
	return std::make_tuple(max_error, std::sqrt(diff_norm2 / base_norm2));
}

template <class A_T, class B_T, class REF_T>
std::tuple<double, double> max_relative_error_and_residual_AxB(
		const unsigned M, const unsigned N, const unsigned K,
		const major_t a_major, const major_t b_major, const major_t r_major,
		const A_T*   const a_ptr, const unsigned lda,
		const B_T*   const b_ptr, const unsigned ldb,
		const REF_T* const r_ptr, const unsigned ldr
		) {
	double max_error = 0.0;
	double max_element = 0.0;
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
				const auto diff = std::abs(r - c);
				max_error = std::max(std::abs(max_error), diff);
				max_element = std::max(std::abs(max_element), c);
				base_norm2 += c * c;
				diff_norm2 += diff * diff;
			});
	return std::make_tuple(max_error / max_element, std::sqrt(diff_norm2 / base_norm2));
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
#pragma omp parallel for collapse(2) reduction(+: base_norm2) reduction(+: diff_norm2)
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
			base_norm2 += a * a;
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
#pragma omp parallel for collapse(2) reduction(max: max_error)
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

template <class A_T, class REF_T>
double max_relative_error(
		const unsigned M, const unsigned N,
		const major_t a_major, const major_t r_major,
		const A_T*   const a_ptr, const unsigned lda,
		const REF_T* const r_ptr, const unsigned ldr
		) {
	double max_error = 0.0;
	double max_element = 0.0;
#pragma omp parallel for collapse(2) reduction(max: max_error)
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
			max_element = std::max(max_element, std::abs(r));
		}
	}
	return max_error;
}

template <class A_T, class REF_T>
std::tuple<double, double> max_error_and_residual(
		const unsigned M, const unsigned N,
		const major_t a_major, const major_t r_major,
		const A_T*   const a_ptr, const unsigned lda,
		const REF_T* const r_ptr, const unsigned ldr
		) {
	double base_norm2 = 0.0;
	double diff_norm2 = 0.0;
	double max_error = 0.0;
#pragma omp parallel for collapse(2) reduction(max: max_error) reduction(+: base_norm2) reduction(+: diff_norm2)
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
			diff_norm2 += diff * diff;
			base_norm2 += a * a;
		}
	}
	return std::make_tuple(max_error, std::sqrt(diff_norm2 / base_norm2));
}

template <class A_T, class REF_T>
std::tuple<double, double> max_relative_error_and_residual(
		const unsigned M, const unsigned N,
		const major_t a_major, const major_t r_major,
		const A_T*   const a_ptr, const unsigned lda,
		const REF_T* const r_ptr, const unsigned ldr
		) {
	double base_norm2 = 0.0;
	double diff_norm2 = 0.0;
	double max_error = 0.0;
	double max_element = 0.0;
#pragma omp parallel for collapse(2) reduction(max: max_error) reduction(+: base_norm2) reduction(+: diff_norm2)
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
			max_element = std::max(max_element, std::abs(r));
			diff_norm2 += diff * diff;
			base_norm2 += a * a;
		}
	}
	return std::make_tuple(max_error / max_element, std::sqrt(diff_norm2 / base_norm2));
}

} // namespace mateval
} // namespace mtk
#endif
