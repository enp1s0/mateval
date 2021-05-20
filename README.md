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
```

- Calculate a condition number
```cpp
// g++ -I/path/to/mateval/include ...
#include <mateval/cond.hpp>
const auto a_major = mtk::mateval::col_major;

const auto cond = mtk::mateval::cond(
    m, n, mtk::mateval::col_major,
    mat_a, m, mtk::mateval::norm_1
    );
```

## License
MIT
