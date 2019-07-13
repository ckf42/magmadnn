#pragma once

#include <cstdio>
#include "compute/operation.h"
#include "compute/variable.h"
#include "graph/graph.h"

#if defined(_HAS_CUDA_)
#include "cusparse.h"
#endif

namespace magmadnn {
namespace math {

//  implement with cuSPARSE cusparseSpMM
template <typename T>
void spgematmul(T alpha, bool trans_A, sparseMatrix<T> *A, bool trans_B, Tensor<T> *B, T beta, Tensor<T> *C);
}  // namespace math
}  // namespace magmadnn