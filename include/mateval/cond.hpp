#ifndef __MATEVAL_COND_HPP__
#define __MATEVAL_COND_HPP__
#include <lapacke.h>
#include <algorithm>
#include <memory>
#include <vector>

#include "common.hpp"

namespace mtk {
namespace mateval {

template <class T>
double cond(
		const unsigned m, const unsigned n,
		const mtk::mateval::major_t a_major,
		const T* const a_ptr, const unsigned lda,
		double* const dp_working_memory,
		const double zero_threshold_ratio = 1e-14
		) {
	// dp_a is a m x n col major fouble precision matrix
	auto dp_a_ptr = dp_working_memory;
	std::vector<double> singular_values(std::min(m, n));

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

	const auto a_norm = LAPACKE_dlange(LAPACK_COL_MAJOR, '1', m, n, dp_a_ptr, m);
	double c;
	const auto res = LAPACKE_dgecon(LAPACK_COL_MAJOR, '1', std::min(m, n), dp_a_ptr, m, a_norm, &c);

	return c;
}

inline unsigned get_cond_working_mem_size(const unsigned m, const unsigned n) {
	return m * n; // for dp_a
}

template <class T>
double cond(
		const unsigned m, const unsigned n,
		const mtk::mateval::major_t a_major,
		const T* const a_ptr, const unsigned lda,
		const double zero_threshold_ratio = 1e-14
		) {
	std::unique_ptr<double[]> working_memory(new double [get_cond_working_mem_size(m, n)]);
	const auto c = cond(
			m, n,
			a_major,
			a_ptr, lda,
			working_memory.get(),
			zero_threshold_ratio
			);
	return c;
}

} // namespace mateval
} // namespace mtk

#endif
