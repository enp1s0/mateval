#include <iostream>
#include <mateval/quad.hpp>

int main() {
	mtk::mateval::quad_t a = 1 / 3.0;
	mtk::mateval::quad_t b = 1 / 3.0;
	mtk::mateval::quad_t c = 1 / 3.0;

	const auto d = (a - b - c) * a;

	std::printf("%e\n", static_cast<double>(d));
}
