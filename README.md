# mateval - tiny matrix evaluation library

## Example

- Evaluate the accuracy of matrix product result
```cpp
// g++ -I/path/to/mateval/include ...
#include <mateval/comparison.hpp>
const auto a_major = mtk::mateval::col_major;
const auto b_major = mtk::mateval::col_major;
const auto r_major = mtk::mateval::col_major;

const std::unordered_map<mtk::mateval::error_t, double> result = mtk::mateval::get_error_AxB(
    mtk::mateval::relative_residual | mtk::mateval::max_relative_error,
    M, N, K,
    a_major, b_major, r_major,
    mat_a, lda,
    mat_b, ldb,
    mat_ref, ldr
    );
const auto residual = result.at(mtk::mateval::relative_residual);
const auto max_relative_error = result.at(mtk::mateval::max_relative_error);
```

- Calculate a condition number
```cpp
// g++ -I/path/to/mateval/include ... -llapacke -llapack -lblas -lgfortran
#include <mateval/cond.hpp>

const auto cond = mtk::mateval::cond(
    m, n, mtk::mateval::col_major,
    mat_a, m, mtk::mateval::norm_1
    );
```

- Calculate an orthogonality
```cpp
// g++ -I/path/to/mateval/include ...
#include <mateval/comparison.hpp>

// Compute |I - At x A| / sqrt(N)
const auto cond = mtk::mateval::orthogonality(
    m, n, mtk::mateval::col_major,
    mat_a, m
    );
```

## CUDA Extensions
1. Build the CUDA extension library
```bash
mkdir build
cd build
cmake ..
make -j8
```

2. Link to your application
```cpp
// nvcc -I/path/to/mateval/include -L/path/to/libmateval_cuda.a -lmateval_cuda ...
#include <mateval/cuda/comparison.hpp>
const auto a_major = mtk::mateval::col_major;
const auto b_major = mtk::mateval::col_major;
const auto r_major = mtk::mateval::col_major;

const std::unordered_map<mtk::mateval::error_t, double> result = mtk::mateval::cuda::get_error_AxB(
    mtk::mateval::relative_residual | mtk::mateval::max_relative_error,
    M, N, K,
    a_major, b_major, r_major,
    mat_a, lda,
    mat_b, ldb,
    mat_ref, ldr
    );
const auto residual = result.at(mtk::mateval::relative_residual);
const auto max_relative_error = result.at(mtk::mateval::max_relative_error);
```

### latms for CUDA

Generate a matrix with specified singular values using QR factorization

See [test code](./tests/cuda_latms.cu)

Compiler option: `-lmateval_cuda_latms`

## License
MIT
