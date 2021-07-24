#include <mateval/latms_cuda.hpp>

template <class T>
void mtk::mateval::latms(
		T* const dst_ptr,
		const unsigned m,
		const unsigned n,
		T* const d,
		const unsigned rank,
		const unsigned seed,
		T* const work_ptr,
		cudaStream_t cuda_stream
		) {
	// Gen U
	// geqrf & ormqr
}
