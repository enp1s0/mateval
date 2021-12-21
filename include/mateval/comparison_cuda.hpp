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
double max_relative_error_AxB(
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

template <class A_T, class B_T, class REF_T>
std::tuple<double, double> max_relative_error_and_residual_AxB(
		const unsigned M, const unsigned N, const unsigned K,
		const major_t a_major, const major_t b_major, const major_t r_major,
		const A_T*   const a_ptr, const unsigned lda,
		const B_T*   const b_ptr, const unsigned ldb,
		const REF_T* const r_ptr, const unsigned ldr
		);

// SVD
template <class U_T, class S_T, class V_T, class REF_T>
double residual_UxSxVt(
		const unsigned M, const unsigned N, const unsigned K,
		const major_t u_major, const major_t v_major, const major_t r_major,
		const U_T*   const u_ptr, const unsigned ldu,
		const S_T*   const s_ptr,
		const V_T*   const v_ptr, const unsigned ldv,
		const REF_T* const r_ptr, const unsigned ldr
		);

// Orthogonality
template <class T>
double orthogonality(
		const unsigned M, const unsigned N,
		const major_t major,
		const T* const ptr, const unsigned ld
		);

} // namespace cuda
} // namespace mateval
} // namespace mtk
#endif
