#ifndef __MATEVAL_COMMON_HPP__
#define __MATEVAL_COMMON_HPP__
#include <vector>
#include <numeric>
#include <functional>
#include <unordered_map>
#include "quad.hpp"

namespace mtk {
namespace mateval {

enum layout_t {
	col_major = 0,
	row_major = 1,
	conj = 2,
};

inline layout_t inv_major(const layout_t m) {
	if (m == col_major) return row_major;
	else return col_major;
}

using error_t = unsigned;
constexpr error_t relative_residual = 0x001;
constexpr error_t max_relative_error = 0x002;
constexpr error_t max_absolute_error = 0x004;
constexpr error_t forward_error = 0x008;
constexpr error_t avg_relative_error = 0x010;

using error_map_t = std::unordered_map<mtk::mateval::error_t, double>;

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
template <class T>
struct accumulate_t {using type = double;};
template <>
struct accumulate_t<double> {using type = quad_t;};

} // namespace mateval
} // namespace mtk
#endif
