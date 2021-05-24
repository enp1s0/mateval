#ifndef __MATEVAL_COMPARISON_CUDA_HPP__
#define __MATEVAL_COMPARISON_CUDA_HPP__
#include <utility>
#include <algorithm>
#include <tuple>
#include "common.hpp"

namespace mtk {
namespace mateval {
namespace cuda {

template <class A_T, class B_T, class REF_T>
double residual_AxB(
		const unsigned M, const unsigned N, const unsigned K,
		const major_t a_major, const major_t b_major, const major_t r_major,
		const A_T*   const a_ptr, const unsigned lda,
		const B_T*   const b_ptr, const unsigned ldb,
		const REF_T* const r_ptr, const unsigned ldr
		);

template <class A_T, class B_T, class REF_T>
double max_error_AxB(
		const unsigned M, const unsigned N, const unsigned K,
		const major_t a_major, const major_t b_major, const major_t r_major,
		const A_T*   const a_ptr, const unsigned lda,
		const B_T*   const b_ptr, const unsigned ldb,
		const REF_T* const r_ptr, const unsigned ldr
		);

template <class A_T, class B_T, class REF_T>
std::tuple<double, double> max_error_and_residual_AxB(
		const unsigned M, const unsigned N, const unsigned K,
		const major_t a_major, const major_t b_major, const major_t r_major,
		const A_T*   const a_ptr, const unsigned lda,
		const B_T*   const b_ptr, const unsigned ldb,
		const REF_T* const r_ptr, const unsigned ldr
		);
} // namespace cuda
} // namespace mateval
} // namespace mtk
#endif
