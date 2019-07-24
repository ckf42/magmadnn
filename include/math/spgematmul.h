#pragma once

#include <cassert>
#include <cstdio>
#include "compute/operation.h"
#include "compute/variable.h"
#include "sparseMatrix/sparseMatrix.h"
#include "utilities_internal.h"
#include "math/matmul.h"

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
                T beta, spMatrix::spMatrix_DENSE<T>* C, void* settings = nullptr, bool col_major_output = false);

#if defined(_HAS_CUDA_)
#if (CUDART_VERSION >= 100100)
struct spgemm_cusparse_settings {
    cusparseSpMMAlg_t algo;
    void* workspace;
    size_t workspace_size;
};
//  implement with cuSPARSE cusparseSpMM
//  assume settings.workspace holds enough space for computation
template <typename T>
void spgematmul_cusparse(T alpha, bool trans_A, spMatrix::sparseMatrix<T>* A, bool trans_B,
                         spMatrix::cusparseSpMatrix_DENSE<T>* B, T beta, spMatrix::cusparseSpMatrix_DENSE<T>* C,
                         spgemm_cusparse_settings settings, bool col_major_output = false);
#elif (CUDART_VERSION < 100100)
//  implement with cuSPARSE cusparse<t>csrmm2
//  by the limitation of cusparse, when trans_B is false, trans_A must be false
template <typename T>
void spgematmul_cusparse_csr(T alpha, bool trans_A, spMatrix::cusparseSpMatrix_CSR<T>* A, bool trans_B,
                             spMatrix::cusparseSpMatrix_DENSE<T>* B, T beta, spMatrix::cusparseSpMatrix_DENSE<T>* C, bool col_major_output = false);
#endif
#endif

}  // namespace math
}  // namespace magmadnn
