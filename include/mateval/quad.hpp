#ifndef __MATEVAL_QUAD_HPP__
#define __MATEVAL_QUAD_HPP__

#ifdef __CUDA_ARCH__
#define CUDA_DEVICE_HOST __device__ __host__
#else
#define CUDA_DEVICE_HOST
#endif

namespace mtk {
namespace mateval {
struct quad_t {
	double high, low;

	CUDA_DEVICE_HOST quad_t(const quad_t& a) {high = a.high;low = a.low;}
	CUDA_DEVICE_HOST quad_t(const double& a) {high = a;low = 0;}
	CUDA_DEVICE_HOST quad_t(const double h, const double l){high = h; low = l;}
	CUDA_DEVICE_HOST quad_t() {}

	CUDA_DEVICE_HOST operator double() const {
		return high + low;
	}
	CUDA_DEVICE_HOST operator float() const {
		return static_cast<double>(*this);
	}
};

CUDA_DEVICE_HOST inline quad_t add(
		const quad_t& a,
		const quad_t& b
		) {
	quad_t c;
	const auto v0 = a.high + b.high;
	const auto v1 = v0 - a.high;
	const auto v2 = (a.high - (v0 - v1)) + (b.high - v1);
	const auto v3 = v2 + a.low + b.low;
	c.high = v0 + v3;
	c.low = v3 - (c.high - v0);

	return c;
}

CUDA_DEVICE_HOST inline quad_t mul(
		const quad_t& a,
		const quad_t& b
		) {
	quad_t c;
	c.high = a.high * b.high;
	const auto v0 = ((1u << 27) + 1) * a.high;
	const auto v1 = v0 - (v0 - a.high);
	const auto v2 = a.high - v1;
	const auto v3 = ((1u << 27) + 1) * b.high;
	const auto v4 = v3 - (v3 - b.high);
	const auto v5 = b.high - v4;
	const auto v6 = ((v1 * v4 - c.high) + v1 * v5 + v2 * v4) + v5 * v2;
	const auto v7 = v6 + (a.high * b.low + a.low * b.high);
	const auto v8 = c.high + v7;
	c.low = v7 - (v8 - c.high);
	c.high = v8;

	return c;
}

CUDA_DEVICE_HOST inline quad_t operator+(
		const quad_t& a,
		const quad_t& b
		) {
	return add(a, b);
}

CUDA_DEVICE_HOST inline quad_t operator+=(
		quad_t& a,
		const quad_t& b
		) {
	return a = add(a, b);
}

CUDA_DEVICE_HOST inline quad_t operator*(
		const quad_t& a,
		const quad_t& b
		) {
	return mul(a, b);
}

CUDA_DEVICE_HOST inline quad_t operator-(
		const quad_t& a
		) {
	return quad_t{-a.high, -a.low};
}

CUDA_DEVICE_HOST inline quad_t operator-(
		const quad_t& a,
		const quad_t& b
		) {
	return add(a, -b);
}

CUDA_DEVICE_HOST inline quad_t abs(
		const quad_t& a
		) {
	if (a.high > 0) {
		return a;
	}
	return -a;
}
} // namespace mateval
} // namespace mtk
#endif
