#ifndef __MATEVAL_CUDA_UTILS__
#define __MATEVAL_CUDA_UTILS__
#include <cublas_v2.h>
#include "../common.hpp"
namespace mtk {
namespace mateval {
namespace utils {
inline cublasOperation_t get_cublas_operation(
		const mtk::mateval::layout_t layout
		) {
	if (layout == mtk::mateval::col_major) {
		return CUBLAS_OP_N;
	}
	return CUBLAS_OP_T;
}

inline mtk::mateval::layout_t get_mateval_layout(
		const cublasOperation_t op
		) {
	if (op == CUBLAS_OP_N) {
		return mtk::mateval::col_major;
	}
	return mtk::mateval::row_major;
}
} // namespace utils
} // namespace mateval
} // namespace mtk
#endif
