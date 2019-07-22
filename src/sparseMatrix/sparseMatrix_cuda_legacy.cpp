#include "sparseMatrix/sparseMatrix_cuda_legacy.h"

namespace magmadnn {
namespace spMatrix {
#if defined(_HAS_CUDA_)
#if defined(USE_CUSPARSE_OLD_API)
//  for concrete class cusparseSpMatrix_DENSE_LEGACY
template <typename T>
cusparseSpMatrix_DENSE_LEGACY<T>::cusparseSpMatrix_DENSE_LEGACY(const Tensor<T>* spMatrixTensorPtr, memory_t mem_type,
                                                                bool copy)
    : spMatrix_DENSE<T>(spMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE, copy) {
    assert(mem_type != HOST);
}
template <typename T>
cusparseSpMatrix_DENSE_LEGACY<T>::cusparseSpMatrix_DENSE_LEGACY(const std::vector<T>& diag, memory_t mem_type)
    : spMatrix_DENSE<T>(diag, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE) {
    assert(mem_type != HOST);
}
template <typename T>
cusparseSpMatrix_DENSE_LEGACY<T>::cusparseSpMatrix_DENSE_LEGACY(unsigned dim0, unsigned dim1, memory_t mem_type)
    : spMatrix_DENSE<T>(dim0, dim1, mem_type, SPARSEMATRIX_FORMAT_HOST_DENSE) {
    assert(mem_type != HOST);
}
template <typename T>
cusparseSpMatrix_DENSE_LEGACY<T>::~cusparseSpMatrix_DENSE_LEGACY(void) {
    /* empty */
}
//  explicit instantiation for type int, float, double
template class cusparseSpMatrix_DENSE_LEGACY<int>;
template class cusparseSpMatrix_DENSE_LEGACY<float>;
template class cusparseSpMatrix_DENSE_LEGACY<double>;

//  for concrete class cusparseSpMatrix_CSR_LEGACY
template <typename T>
void cusparseSpMatrix_CSR_LEGACY<T>::createDesc(void) {
    this->_descriptor = new cusparseMatDescr_t;
    cusparseErrchk(cusparseCreateMatDescr(AS_TYPE(cusparseMatDescr_t*, this->_descriptor)));
    cusparseErrchk(cusparseSetMatIndexBase(*AS_TYPE(cusparseMatDescr_t*, this->_descriptor), CUSPARSE_INDEX_BASE_ZERO));
    cusparseErrchk(cusparseSetMatType(*AS_TYPE(cusparseMatDescr_t*, this->_descriptor), CUSPARSE_MATRIX_TYPE_GENERAL));
    this->_descriptor_is_set = true;
}
template <typename T>
cusparseSpMatrix_CSR_LEGACY<T>::~cusparseSpMatrix_CSR_LEGACY(void) {
    if (this->_descriptor_is_set) {
        cusparseErrchk(cusparseDestroyMatDescr(*AS_TYPE(cusparseMatDescr_t*, this->_descriptor)));
        this->_descriptor_is_set = false;
    }
}
template <typename T>
cusparseSpMatrix_CSR_LEGACY<T>::cusparseSpMatrix_CSR_LEGACY(const Tensor<T>* spMatrixTensorPtr, memory_t mem_type,
                                                            bool copy)
    : spMatrix_CSR<T>(spMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE) {
    assert(mem_type != HOST);
    createDesc();
}
template <typename T>
cusparseSpMatrix_CSR_LEGACY<T>::cusparseSpMatrix_CSR_LEGACY(const std::vector<T>& diag, memory_t mem_type)
    : spMatrix_CSR<T>(diag, mem_type) {
    assert(mem_type != HOST);
    createDesc();
}
template <typename T>
cusparseSpMatrix_CSR_LEGACY<T>::cusparseSpMatrix_CSR_LEGACY(unsigned dim0, unsigned dim1, const std::vector<T>& valList,
                                                            const std::vector<int>& rowAccum,
                                                            const std::vector<int>& colIdx, memory_t mem_type)
    : spMatrix_CSR<T>(dim0, dim1, valList, rowAccum, colIdx, mem_type) {
    assert(mem_type != HOST);
    createDesc();
}
//template <typename T>
//void cusparseSpMatrix_CSR_LEGACY<T>::setMatType(cusparseMatrixType_t type) {
//    cusparseErrchk(cusparseSetMatType(*AS_TYPE(cusparseMatDescr_t*, this->_descriptor), type));
//}

#endif
#endif
}  //  namespace spMatrix
}  //  namespace magmadnn