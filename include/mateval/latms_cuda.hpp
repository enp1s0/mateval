#ifndef __MATEVAL_LATMS_CUDA_HPP__
#define __MATEVAL_LATMS_CUDA_HPP__
#include <cstdint>

namespace mtk {
namespace mateval {
namespace cuda {
// working memory space
static const int device_memory = 1;
static const int host_memory = 2;

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
		T* const dst_ptr,
		const unsigned m,
		const unsigned n,
		T* const d,
		const unsigned rank,
		const unsigned long long seed,
		const int working_memory_type = device_memory,
		cudaStream_t cuda_stream = 0
		);
} // namespace cuda
} // namespace mateval
} // namespace mtk
#endif

