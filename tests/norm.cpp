#include <iostream>
#include <vector>
#include <mateval/norm.hpp>

namespace {
template <class T>
const char* get_name();
template <> const char* get_name<float >() {return "float" ;}
template <> const char* get_name<double>() {return "double";}
} // noname namespace

template <class T>
void eval(
	const std::size_t len
	) {
	std::vector<T> array(len);
	for (std::size_t i = 0; i < len; i++) {
		array[i] = i + 1;
	}

	const auto norm = mtk::mateval::norm(array.data(), len);
	const auto ref_norm = std::sqrt(len * (len + 1) * (2 * len + 1) / 6);

	std::printf("[dtype = %6s, len = %lu] norm = %e, ref = %e, %s\n",
							get_name<T>(), len,
							norm, ref_norm,
							std::abs(norm - ref_norm) < ref_norm * 1e-4 ? "OK" : "NG");
}

int main() {
	eval<float >(1lu << 10);
	eval<double>(1lu << 10);
	eval<float >(1lu << 20);
	eval<double>(1lu << 20);
}
