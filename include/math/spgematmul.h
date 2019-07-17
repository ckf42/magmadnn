#pragma once

#include <cassert>
#include <cstdio>
#include "compute/operation.h"
#include "compute/variable.h"
#include "sparseMatrix/sparseMatrix.h"
#include "utilities_internal.h"

#if defined(_HAS_CUDA_)
#include "cusparse.h"
#endif

namespace magmadnn {
namespace math {

//  compute C = alpha*op(A)*op(B)+beta*C
//  B, C must be of dense class
//  determine which routine to use by format of A
template <typename T>
void spgematmul(T alpha, bool trans_A, spMatrix::sparseMatrix<T>* A, bool trans_B, spMatrix::spMatrix_DENSE<T>* B,
                T beta, spMatrix::spMatrix_DENSE<T>* C, void* settings = nullptr);

#if defined(_HAS_CUDA_)
struct spgemm_cusparse_settings {
    cusparseSpMMAlg_t algo;
    void* workspace;
    size_t workspace_size;
};
//  implement with cuSPARSE cusparseSpMM
//  assume settings.workspace holds enough space for computation
template <typename T>
void spgematmul_cusparse(T alpha, bool trans_A, spMatrix::cusparseSpMatrix_CSR<T>* A, bool trans_B,
                         spMatrix::cusparseSpMatrix_DENSE<T>* B, T beta, spMatrix::cusparseSpMatrix_DENSE<T>* C,
                         spgemm_cusparse_settings settings);

#endif

}  // namespace math
}  // namespace magmadnn
