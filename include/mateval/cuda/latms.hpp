#ifndef __MATEVAL_LATMS_CUDA_HPP__
#define __MATEVAL_LATMS_CUDA_HPP__
#include <cstdint>
#include "../common.hpp"

namespace mtk {
namespace mateval {
namespace cuda {
// working memory space
enum memory_type {
	device_memory = 1,
	host_memory = 2
};

//             +-------------+
//             |             |
//             |      V      |
//             |             |
//       rank  +-------------+
//      +-----+
//      |     |
//  rank|  S  |
//      |     |
//      +-----+       n
// +---+       +-------------+
// |   |       |             |
// |   |       |             |
// | U |      m|      A      |
// |   |       |             |
// |   |       |             |
// +---+       +-------------+
template <class T>
void latms(
		const unsigned m,
		const unsigned n,
		const mtk::mateval::layout_t major,
		T* const dst_ptr,
		const unsigned ldm,
		T* const d,
		const unsigned rank,
		const unsigned long long seed,
		const memory_type working_memory_type = device_memory,
		cudaStream_t cuda_stream = 0
		);
} // namespace cuda
} // namespace mateval
} // namespace mtk
#endif
