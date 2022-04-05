#ifndef __MATEVAL_COND_HPP__
#define __MATEVAL_COND_HPP__
#include <lapacke.h>
#include <algorithm>
#include <memory>
#include <vector>

#include "common.hpp"

namespace mtk {
namespace mateval {

static const char norm_1 = '1';
static const char norm_infinity = 'I';

template <class T>
double cond(
		const unsigned m, const unsigned n,
		const mtk::mateval::layout_t a_major,
		const T* const a_ptr, const unsigned lda,
		double* const dp_working_memory,
		const char norm_mode = mtk::mateval::norm_1
		) {
	// dp_a is a m x n col major fouble precision matrix
	auto dp_a_ptr = dp_working_memory;

	// copy the input matrix a to dp_a
	for (unsigned i = 0; i < n; i++) {
		for (unsigned j = 0; j < m; j++) {
			unsigned src_index;
			if (a_major == mtk::mateval::col_major) {
				src_index = j + i * lda;
			} else {
				src_index = j * lda + i;
			}
			dp_a_ptr[j + i * m] = static_cast<double>(a_ptr[src_index]);
		}
	}

	const auto a_norm = LAPACKE_dlange(LAPACK_COL_MAJOR, norm_mode, m, n, dp_a_ptr, m);
	auto ipiv_uptr = std::unique_ptr<int[]>(new int[m]);
	LAPACKE_dgetrf(LAPACK_COL_MAJOR, m, n, dp_a_ptr, m, ipiv_uptr.get());
	double c;
	const auto res = LAPACKE_dgecon(LAPACK_COL_MAJOR, norm_mode, std::min(m, n), dp_a_ptr, m, a_norm, &c);

	return 1. / c;
}

inline unsigned get_cond_working_mem_size(const unsigned m, const unsigned n) {
	return m * n; // for dp_a
}

template <class T>
double cond(
		const unsigned m, const unsigned n,
		const mtk::mateval::layout_t a_major,
		const T* const a_ptr, const unsigned lda,
		const char norm_mode = mtk::mateval::norm_1
		) {
	std::unique_ptr<double[]> working_memory(new double [get_cond_working_mem_size(m, n)]);
	const auto c = cond(
			m, n,
			a_major,
			a_ptr, lda,
			working_memory.get(),
			norm_mode
			);
	return c;
}

} // namespace mateval
} // namespace mtk

#endif
