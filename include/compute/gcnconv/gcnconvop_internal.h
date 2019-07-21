#pragma once

#include "math/spgematmul.h"
#include "sparseMatrix/sparseMatrix.h"

namespace magmadnn {
namespace internal {

#if defined(_HAS_CUDA_)
template <typename T>
inline void set_cusparse_spmm_settings(void*& settings, cudaDataType data_type, const T* alpha, bool spMatDoTrans,
                                       spMatrix::sparseMatrix<T>* spMat, bool dnMatDoTrans,
                                       spMatrix::spMatrix_DENSE<T>* dnMat, const T* beta,
                                       spMatrix::spMatrix_DENSE<T>* dnOut, cusparseSpMMAlg_t alg);
#endif

}  //  namespace internal
}  //  namespace magmadnn
