#ifndef __MATEVAL_LATMS_CUDA_HPP__
#define __MATEVAL_LATMS_CUDA_HPP__
#include <cstdint>
#include <curand.h>
#include <cublas.h>
#include <cublas_v2.h>
#include <cusolver.h>

namespace mtk {
namespace mateval {
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
		const unsigned seed,
		T* const work_ptr,
		cudaStream_t cuda_stream = 0
		);
} // namespace mateval
} // namespace mtk
#endif

