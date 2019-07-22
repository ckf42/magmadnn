#pragma once

#include "sparseMatrix/sparseMatrix.h"

namespace magmadnn {
namespace spMatrix {
#if defined(_HAS_CUDA_)
#if defined(USE_CUSPARSE_OLD_API)

//  Concrete class for sparse matrix in cusparse dense format in GPU memory
//  For dense matrix used in CUDA version below 10.1
//  wrapper for tensor on device
//  Since cusparse takes column-major dense matrix but Tensor stores in row-major format, internal _data are stored as
//  column-major. Hense whenever a object is uesd in cusparse, an extra transpose and a reshape will be needed
template <typename T>
class cusparseSpMatrix_DENSE_LEGACY : public spMatrix_DENSE<T> {
   private:
    cusparseSpMatrix_DENSE_LEGACY(const cusparseSpMatrix_DENSE_LEGACY<T>& that) = delete;
    cusparseSpMatrix_DENSE_LEGACY<T>& operator=(const cusparseSpMatrix_DENSE_LEGACY<T>& that) = delete;

   public:
    cusparseSpMatrix_DENSE_LEGACY(const Tensor<T>* spMatrixTensorPtr, memory_t mem_type = MANAGED, bool copy = false);
    cusparseSpMatrix_DENSE_LEGACY(const std::vector<T>& diag, memory_t mem_type = MANAGED);
    cusparseSpMatrix_DENSE_LEGACY(unsigned dim0, unsigned dim1, memory_t mem_type = MANAGED);
    ~cusparseSpMatrix_DENSE_LEGACY(void);
};

//  concrete class for sparse matrix in cusparse CSR format
template <typename T>
class cusparseSpMatrix_CSR_LEGACY : public spMatrix_CSR<T> {
   private:
    cusparseSpMatrix_CSR_LEGACY(const cusparseSpMatrix_CSR_LEGACY<T>& that) = delete;
    cusparseSpMatrix_CSR_LEGACY<T>& operator=(const cusparseSpMatrix_CSR_LEGACY<T>& that) = delete;
    void createDesc(void);
   public:
    cusparseSpMatrix_CSR_LEGACY(const Tensor<T>* spMatrixTensorPtr, memory_t mem_type = MANAGED, bool copy = false);
    cusparseSpMatrix_CSR_LEGACY(const std::vector<T>& diag, memory_t mem_type = MANAGED);
    cusparseSpMatrix_CSR_LEGACY(unsigned dim0, unsigned dim1, const std::vector<T>& valList,
                                const std::vector<int>& rowAccum, const std::vector<int>& colIdx,
                                memory_t mem_type = MANAGED);
    ~cusparseSpMatrix_CSR_LEGACY(void);
	//  in case it is needed
    //inline void setMatType(cusparseMatrixType_t type);
};

#endif
#endif
}  //  namespace spMatrix
}  //  namespace magmadnn