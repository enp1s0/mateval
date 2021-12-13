#include <iostream>
#include <vector>
#include <mateval/common.hpp>

constexpr unsigned N = 100;

int main() {
	std::vector<double> array(N);
	for (unsigned i = 0; i < N; i++) {
		array[i] = i + 1;
	}

	const auto [mean, var] = mtk::mateval::utils::calc_mean_and_var(array);

	const auto expected_mean = (N + 1) / 2.;
	const auto expected_var  = N * (N + 1) / 12.;

	std::printf("mean : %e (expected: %e) [%s]\n", mean, expected_mean, (std::abs((mean - expected_mean) / expected_mean) < 1e-14) ? "PASSED" : "FAILED");
	std::printf("var  : %e (expected: %e) [%s]\n", var , expected_var , (std::abs((var  - expected_var ) / expected_var ) < 1e-14) ? "PASSED" : "FAILED");
}
