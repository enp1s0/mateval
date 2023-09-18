#pragma once
#include <cstdint>

namespace mtk {
namespace mateval {
namespace cuda {
template <class T>
double norm(
		const T* const ptr,
		const std::size_t len
		);
} // namespace cuda
} // namespace mateval
} // namespace mtk
