#pragma once
#include <cmath>
#include <omp.h>
#include "common.hpp"

namespace mtk {
namespace mateval {
template <class T>
double norm(
		const T* const ptr,
		const std::size_t len
		) {
	using acc_t = typename mtk::mateval::accumulate_t<T>::type;

	acc_t sum(0);
#pragma omp parallel
	{
		acc_t local_sum(0);
		for (std::size_t i = omp_get_thread_num(); i < len; i += omp_get_num_threads()) {
			const acc_t a(ptr[i]);
			local_sum = local_sum + a * a;
		}
#pragma omp critical
		sum += local_sum;
	}

	const double sum_fp64 = sum;

	return std::sqrt(sum_fp64);
}
} // namespace mateval
} // namespace mtk
