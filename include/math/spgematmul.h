#pragma once

#include <cassert>
#include <cstdio>
#include "sparseMatrix/sparseMatrix.h"
#include "compute/operation.h"
#include "compute/variable.h"
#include "graph/graph.h"
#include "utilities_internal.h"

#if defined(_HAS_CUDA_)
#include "cusparse.h"
#endif

namespace magmadnn {
namespace math {

template <typename T>
void spgematmul(bool trans_A, spMatrix::sparseMatrix<T>* A, bool trans_B, spMatrix::spMatrix_DENSE<T>* B,
                spMatrix::spMatrix_DENSE<T>* C);

#if defined(_HAS_CUDA_)
struct spgemm_cusparse_settings {
    cusparseSpMMAlg_t algo;
    void* workspace;
    size_t workspace_size;
};

//  implement with cuSPARSE cusparseSpMM
//  assume settings.workspace holds enough space for computation
template <typename T>
void spgematmul_cusparse(bool trans_A, spMatrix::cusparseSpMatrix_CSR<T>* A, bool trans_B,
                         spMatrix::cusparseSpMatrix_DENSE<T>* B, spMatrix::cusparseSpMatrix_DENSE<T>* C,
                         spgemm_cusparse_settings settings);

#endif

}  // namespace math
}  // namespace magmadnn
