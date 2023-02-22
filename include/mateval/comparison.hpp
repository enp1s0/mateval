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
		const layout_t a_major, const layout_t b_major,
		const A_T* const a_ptr, const unsigned lda,
		const B_T* const b_ptr, const unsigned ldb,
		const Func func
		) {
	using acc_t = typename mtk::mateval::accumulate_t<B_T>::type;
#pragma omp parallel for collapse(2)
	for (unsigned m = 0; m < M; m++) {
		for (unsigned n = 0; n < N; n++) {
			acc_t c = 0.0;
			for (unsigned k = 0; k < K; k++) {
				// load A
				acc_t a;
				if (a_major == col_major) {
					a = a_ptr[k * lda + m];
				} else {
					a = a_ptr[m * lda + k];
				}

				// load B
				acc_t b;
				if (b_major == col_major) {
					b = b_ptr[k + ldb * n];
				} else {
					b = b_ptr[k * ldb + n];
				}
				c += a * b;
			}
#pragma omp critical
			{func(static_cast<double>(c), m, n);}
		}
	}
}

template <class A_T, class B_T, class Func>
void foreach_AxB_with_abs(
		const unsigned M, const unsigned N, const unsigned K,
		const layout_t a_major, const layout_t b_major,
		const A_T* const a_ptr, const unsigned lda,
		const B_T* const b_ptr, const unsigned ldb,
		const Func func
		) {
	using acc_t = typename mtk::mateval::accumulate_t<B_T>::type;
#pragma omp parallel for collapse(2)
	for (unsigned m = 0; m < M; m++) {
		for (unsigned n = 0; n < N; n++) {
			acc_t c = 0.0;
			acc_t abs_c = 0.0;
			for (unsigned k = 0; k < K; k++) {
				// load A
				acc_t a;
				if (a_major == col_major) {
					a = a_ptr[k * lda + m];
				} else {
					a = a_ptr[m * lda + k];
				}

				// load B
				acc_t b;
				if (b_major == col_major) {
					b = b_ptr[k + ldb * n];
				} else {
					b = b_ptr[k * ldb + n];
				}
				c += a * b;
				abs_c += abs(a) * abs(b);
			}
#pragma omp critical
			{func(static_cast<double>(c), static_cast<double>(abs_c), m, n);}
		}
	}
}

template <class A_T, class B_T, class REF_T>
mtk::mateval::error_map_t get_error_AxB(
		const mtk::mateval::error_t error,
		const unsigned M, const unsigned N, const unsigned K,
		const layout_t a_major, const layout_t b_major, const layout_t r_major,
		const A_T*   const a_ptr, const unsigned lda,
		const B_T*   const b_ptr, const unsigned ldb,
		const REF_T* const r_ptr, const unsigned ldr
		) {
	double max_error = 0.0;
	double max_relative_error = 0.0;
	double sum_relative_error = 0.0;
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
				if (error & mtk::mateval::max_absolute_error) {
					max_error = std::max(std::abs(diff), max_error);
				}
				if (error & mtk::mateval::max_relative_error) {
					max_relative_error = std::max(max_relative_error, std::abs(diff / c));
				}
				if (error & mtk::mateval::relative_residual) {
					base_norm2 += c * c;
					diff_norm2 += diff * diff;
				}
				if (error & mtk::mateval::avg_relative_error) {
					if (r != 0) {
						sum_relative_error += diff / std::abs(r);
					}
				}
			});
	mtk::mateval::error_map_t result;
	if (error & mtk::mateval::relative_residual) {
		result.insert(std::make_pair(mtk::mateval::relative_residual, std::sqrt(diff_norm2 / base_norm2)));
	}
	if (error & mtk::mateval::max_relative_error) {
		result.insert(std::make_pair(mtk::mateval::max_relative_error, max_relative_error));
	}
	if (error & mtk::mateval::max_absolute_error) {
		result.insert(std::make_pair(mtk::mateval::max_absolute_error, max_error));
	}
	if (error & mtk::mateval::avg_relative_error) {
		result.insert(std::make_pair(mtk::mateval::avg_relative_error, sum_relative_error / (M * N)));
	}
	return result;
}

template <class A_T, class REF_T>
mtk::mateval::error_map_t get_error(
		const mtk::mateval::error_t error,
		const unsigned M, const unsigned N,
		const layout_t a_major, const layout_t r_major,
		const A_T*   const a_ptr, const unsigned lda,
		const REF_T* const r_ptr, const unsigned ldr
		) {
	double base_norm2 = 0.0;
	double diff_norm2 = 0.0;
	double max_error = 0.0;
	double sum_relative_error = 0.0;
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
			if (error & (mtk::mateval::max_relative_error | mtk::mateval::max_absolute_error)) {
				max_error = std::max(std::abs(diff), max_error);
			}
			if (error & mtk::mateval::max_absolute_error) {
				max_element = std::max(std::abs(r), max_element);
			}
			if (error & mtk::mateval::relative_residual) {
				base_norm2 += r * r;
				diff_norm2 += diff * diff;
			}
			if (error & mtk::mateval::avg_relative_error) {
				if (r != 0) {
					sum_relative_error += diff / std::abs(r);
				}
			}
		}
	}
	mtk::mateval::error_map_t result;
	if (error & mtk::mateval::relative_residual) {
		result.insert(std::make_pair(mtk::mateval::relative_residual, std::sqrt(diff_norm2 / base_norm2)));
	}
	if (error & mtk::mateval::max_relative_error) {
		result.insert(std::make_pair(mtk::mateval::max_relative_error, max_error / max_element));
	}
	if (error & mtk::mateval::max_absolute_error) {
		result.insert(std::make_pair(mtk::mateval::max_absolute_error, max_error));
	}
	if (error & mtk::mateval::avg_relative_error) {
		result.insert(std::make_pair(mtk::mateval::avg_relative_error, sum_relative_error / (M * N)));
	}
	return result;
}

template <class U_T, class S_T, class V_T, class REF_T>
double residual_UxSxVt(
		const unsigned M, const unsigned N, const unsigned K,
		const layout_t u_major, const layout_t v_major, const layout_t r_major,
		const U_T* const u_ptr, const unsigned ldu,
		const S_T* const s_ptr,
		const V_T* const v_ptr, const unsigned ldv,
		const REF_T* const r_ptr, const unsigned ldr
		) {
	double base_norm2 = 0.;
	double diff_norm2 = 0.;
#pragma omp parallel for collapse(2) reduction(+: base_norm2) reduction(+: diff_norm2)
	for (unsigned m = 0; m < M; m++) {
		for (unsigned n = 0; n < N; n++) {
			double c = 0.0;
			for (unsigned k = 0; k < K; k++) {
				// load V
				double u;
				if (u_major == col_major) {
					u = u_ptr[k * ldu + m];
				} else {
					u = u_ptr[m * ldu + k];
				}

				// load V
				double v;
				if (v_major == col_major) {
					v = v_ptr[k * ldv + n];
				} else {
					v = v_ptr[k + ldv * n];
				}

				// load S
				const double s = s_ptr[k];
				c += u * s * v;
			}

			double r;
			if (r_major == mtk::mateval::col_major) {
				r = r_ptr[m + n * ldr];
			} else {
				r = r_ptr[n + m * ldr];
			}
			const auto diff = c - r;
			diff_norm2 += diff * diff;
			base_norm2 += c * c;
		}
	}
	return std::sqrt(diff_norm2 / base_norm2);
}

// Orthogonality
template <class T>
double orthogonality(
		const unsigned M, const unsigned N,
		const layout_t major,
		const T* const ptr, const unsigned ld
		) {
	double sum = 0.;
	foreach_AxB(
			N, N, M,
			inv_major(major), major,
			ptr, ld,
			ptr, ld,
			[&](const double c, const unsigned m, const unsigned n) {
			const auto v = (m == n ? 1. : 0.) - c;
			sum += v * v;
			});
	return std::sqrt(sum / N);
}

} // namespace mateval
} // namespace mtk
#endif
