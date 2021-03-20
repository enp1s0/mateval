# mateval - tiny matrix evaluation library

## Sample
```cpp
// g++ -I/path/to/mateval/include ...
#include <mateval/mateval.hpp>
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

## License
MIT
