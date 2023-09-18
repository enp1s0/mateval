#ifndef __MATEVAL_MINMAX_CUDA_HPP__
#define __MATEVAL_MINMAX_CUDA_HPP__
#include "../common.hpp"

namespace mtk {
namespace mateval {

using operation_t = unsigned;
constexpr operation_t op_max = 0x001;
constexpr operation_t op_min = 0x002;
constexpr operation_t op_abs_max = 0x004;
constexpr operation_t op_abs_min = 0x008;

namespace cuda {
template <class T>
mtk::mateval::error_map_t operate (
	const mtk::mateval::operation_t operation,
	const mtk::mateval::layout_t layout,
	const std::size_t m,
	const std::size_t n,
	const T* const mat_ptr,
	const std::size_t ld
	);
} // namespace cuda
} // namespace mateval
} // namespace mtk
#endif
