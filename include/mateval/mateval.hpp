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
void iterator_AxB(
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
} // namespace mateval
} // namespace mtk
#endif
