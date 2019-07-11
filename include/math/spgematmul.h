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
void spgematmul(T alpha, sparseMatrix<T> &A, bool transpo_A, Tensor<T> *B, bool transpo_B, Tensor<T> *C);
}
}  // namespace magmadnn