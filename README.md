# mateval - tiny matrix evaluation library

## Example

- Evaluate the accuracy of matrix product result
```cpp
// g++ -I/path/to/mateval/include ...
#include <mateval/comparison.hpp>
const auto a_major = mtk::mateval::col_major;
const auto b_major = mtk::mateval::col_major;
const auto r_major = mtk::mateval::col_major;

// Evaluation of residual
const auto residual = mtk::mateval::residual_AxB(
    M, N, K,
    a_major, b_major, r_major,
    mat_a, lda,
    mat_b, ldb,
    mat_ref, ldr
    );

// Evaluation of max_error
const auto max_error = mtk::mateval::max_error_AxB(
    M, N, K,
    a_major, b_major, r_major,
    mat_a, lda,
    mat_b, ldb,
    mat_ref, ldr
    );

// Evaluation of max_relative_error
// max_i,j |C_ref - C_target|_i,j / (|A||B|)_ij
const auto max_error = mtk::mateval::max_relative_error_AxB(
    M, N, K,
    a_major, b_major, r_major,
    mat_a, lda,
    mat_b, ldb,
    mat_ref, ldr
    );
```

- Calculate a condition number
```cpp
// g++ -I/path/to/mateval/include ... -llapacke -llapack -lblas -lgfortran
#include <mateval/cond.hpp>
const auto a_major = mtk::mateval::col_major;

const auto cond = mtk::mateval::cond(
    m, n, mtk::mateval::col_major,
    mat_a, m, mtk::mateval::norm_1
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
// g++ -I/path/to/mateval/include -L/path/to/libmateval_cuda.a -lmateval_cuda ...
#include <mateval/comparison.hpp>
const auto a_major = mtk::mateval::col_major;
const auto b_major = mtk::mateval::col_major;
const auto r_major = mtk::mateval::col_major;

// Evaluation of residual
const auto residual = mtk::mateval::cuda::residual_AxB(
    M, N, K,
    a_major, b_major, r_major,
    mat_a, lda,
    mat_b, ldb,
    mat_ref, ldr
    );

// Evaluation of max_error
const auto max_error = mtk::mateval::cuda::max_error_AxB(
    M, N, K,
    a_major, b_major, r_major,
    mat_a, lda,
    mat_b, ldb,
    mat_ref, ldr
    );
```


## License
MIT
