#ifndef __MATEVAL_COMPARISON_CUDA_HPP__
#define __MATEVAL_COMPARISON_CUDA_HPP__
#include <utility>
#include <algorithm>
#include <unordered_map>
#include "../common.hpp"

namespace mtk {
namespace mateval {
namespace cuda {

template <class A_T, class B_T, class C_T, class REF_T, class ALPHA_T, class BETA_T>
mtk::mateval::error_map_t get_error_GEMM(
		const mtk::mateval::error_t error,
		const unsigned M, const unsigned N, const unsigned K,
		const layout_t a_major, const layout_t b_major, const layout_t c_major, const layout_t r_major,
		const ALPHA_T alpha,
		const A_T*   const a_ptr, const unsigned lda,
		const B_T*   const b_ptr, const unsigned ldb,
		const BETA_T beta,
		const C_T*   const c_ptr, const unsigned ldc,
		const REF_T* const r_ptr, const unsigned ldr
		);

template <class A_T, class B_T, class REF_T>
mtk::mateval::error_map_t get_error_AxB(
		const mtk::mateval::error_t error,
		const unsigned M, const unsigned N, const unsigned K,
		const layout_t a_major, const layout_t b_major, const layout_t r_major,
		const A_T*   const a_ptr, const unsigned lda,
		const B_T*   const b_ptr, const unsigned ldb,
		const REF_T* const r_ptr, const unsigned ldr
		);

template <class A_T, class REF_T>
mtk::mateval::error_map_t get_error(
		const mtk::mateval::error_t error,
		const unsigned M, const unsigned N,
		const layout_t a_major, const layout_t r_major,
		const A_T*   const a_ptr, const unsigned lda,
		const REF_T* const r_ptr, const unsigned ldr
		);

// SVD
template <class U_T, class S_T, class V_T, class REF_T>
double residual_UxSxVt(
		const unsigned M, const unsigned N, const unsigned K,
		const layout_t u_major, const layout_t v_major, const layout_t r_major,
		const U_T*   const u_ptr, const unsigned ldu,
		const S_T*   const s_ptr,
		const V_T*   const v_ptr, const unsigned ldv,
		const REF_T* const r_ptr, const unsigned ldr
		);

// Orthogonality
template <class T>
double orthogonality(
		const unsigned M, const unsigned N,
		const layout_t major,
		const T* const ptr, const unsigned ld
		);

} // namespace cuda
} // namespace mateval
} // namespace mtk
#endif
