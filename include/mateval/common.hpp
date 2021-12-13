#ifndef __MATEVAL_COMMON_HPP__
#define __MATEVAL_COMMON_HPP__
#include <vector>
#include <numeric>
#include <functional>

namespace mtk {
namespace mateval {

enum major_t {
	col_major = 0,
	row_major = 1,
};

namespace utils {
template <class T>
inline std::pair<T, T> calc_mean_and_var(
		const std::vector<T>& array
		) {
	const auto mean = std::accumulate(array.begin(), array.end(), static_cast<T>(0), [&](const T a, const T b){return a + b;}) / array.size();
	const auto var = std::accumulate(array.begin(), array.end(), static_cast<T>(0), [&](const T a, const T b){return a + (b - mean) * (b - mean);}) / (array.size() - 1);

	return std::make_pair(mean, var);
}
} // namespace utils

} // namespace mateval
} // namespace mtk
#endif
