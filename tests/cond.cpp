#include <iostream>
#include <memory>
#include <random>
#include <mateval/cond.hpp>

template <class T>
void cond_dense_test(const unsigned m, const unsigned n, const char norm_mode) {
	auto a_uptr = std::unique_ptr<T[]>(new T[m * n]);

	std::mt19937 mt(0);
	std::uniform_real_distribution<T> dist(-10, 10);

	for (unsigned i = 0; i < m * n; i++) {
		a_uptr.get()[i] = dist(mt);
	}

	const auto cond = mtk::mateval::cond(m, n, mtk::mateval::col_major, a_uptr.get(), m, norm_mode);

	std::printf("m = %u, n = %u, norm = %c, cond = %e\n", m, n, norm_mode, cond);
}

int main() {
	cond_dense_test<float >(1u << 10, 1u << 10, mtk::mateval::norm_1);
	cond_dense_test<double>(1u << 10, 1u << 10, mtk::mateval::norm_1);
	cond_dense_test<float >(1u << 10, 1u << 10, mtk::mateval::norm_infinity);
	cond_dense_test<double>(1u << 10, 1u << 10, mtk::mateval::norm_infinity);
}
