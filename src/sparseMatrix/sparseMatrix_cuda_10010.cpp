#include "sparseMatrix/sparseMatrix_cuda_10010.h"

namespace magmadnn {
namespace spMatrix {
//  todo: clean up
#if defined(_HAS_CUDA_)
#if defined(USE_CUSPARSE_NEW_API)
//  for concrete class cusparseSpMatrix_DENSE_10010
template <typename T>
cusparseSpMatrix_DENSE_10010<T>::cusparseSpMatrix_DENSE_10010(const Tensor<T>* spMatrixTensorPtr, memory_t mem_type,
                                                              bool copy)
    : spMatrix_DENSE<T>(spMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE, false) {
    assert(mem_type != HOST);
    fprintf(stderr, "Requested template type for cusparseSpMatrix_DENSE_10010 is not supported\n");
}
template <typename T>
cusparseSpMatrix_DENSE_10010<T>::cusparseSpMatrix_DENSE_10010(const std::vector<T>& diag, memory_t mem_type)
    : spMatrix_DENSE<T>(diag, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE) {
    assert(mem_type != HOST);
    fprintf(stderr, "Requested template type for cusparseSpMatrix_DENSE_10010 is not supported\n");
}
template <typename T>
cusparseSpMatrix_DENSE_10010<T>::cusparseSpMatrix_DENSE_10010(unsigned dim0, unsigned dim1, memory_t mem_type)
    : spMatrix_DENSE<T>(dim0, dim1, mem_type, SPARSEMATRIX_FORMAT_HOST_DENSE) {
    assert(mem_type != HOST);
    std::fprintf(stderr, "Requested template type for cusparseSpMatrix_DENSE_10010 is not supported\n");
}
template <typename T>
cusparseSpMatrix_DENSE_10010<T>::~cusparseSpMatrix_DENSE_10010(void) {
    if (this->_descriptor_is_set) {
        cusparseDestroyDnMat(*AS_TYPE(cusparseDnMatDescr_t*, this->_descriptor));
        this->_descriptor_is_set = false;
        //  todo: check if desctiptor is also freed (e.g. assigned nullptr)
    }
}
template <typename T>
void cusparseSpMatrix_DENSE_10010<T>::createDesc(cudaDataType_t cuda_data_type) {
    _descriptor = new cusparseDnMatDescr_t;
    cusparseErrchk(cusparseCreateDnMat(AS_TYPE(cusparseDnMatDescr_t*, _descriptor), _dim0, _dim1, _dim0,
                                       _data->get_ptr(), cuda_data_type, CUSPARSE_ORDER_COL));
    _descriptor_is_set = true;
}
//  specialization
template <>
cusparseSpMatrix_DENSE_10010<int>::cusparseSpMatrix_DENSE_10010(const Tensor<int>* spMatrixTensorPtr, memory_t mem_type,
                                                                bool copy)
    : spMatrix_DENSE<int>(spMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE, copy) {
    assert(mem_type != HOST);
    if (copy) {
        internal::transpose_full_device<int>(_data, _data);
    }
    this->createDesc(CUDA_R_32I);
}
template <>
cusparseSpMatrix_DENSE_10010<float>::cusparseSpMatrix_DENSE_10010(const Tensor<float>* spMatrixTensorPtr,
                                                                  memory_t mem_type, bool copy)
    : spMatrix_DENSE<float>(spMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE, copy) {
    assert(mem_type != HOST);
    if (copy) {
        internal::transpose_full_device<float>(_data, _data);
    }
    this->createDesc(CUDA_R_32F);
}
template <>
cusparseSpMatrix_DENSE_10010<double>::cusparseSpMatrix_DENSE_10010(const Tensor<double>* spMatrixTensorPtr,
                                                                   memory_t mem_type, bool copy)
    : spMatrix_DENSE<double>(spMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE, copy) {
    assert(mem_type != HOST);
    if (copy) {
        internal::transpose_full_device<double>(_data, _data);
    }
    this->createDesc(CUDA_R_64F);
}
template <>
cusparseSpMatrix_DENSE_10010<int>::cusparseSpMatrix_DENSE_10010(const std::vector<int>& diag, memory_t mem_type)
    : spMatrix_DENSE<int>(diag, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE) {
    assert(mem_type != HOST);
    this->createDesc(CUDA_R_32I);
}
template <>
cusparseSpMatrix_DENSE_10010<float>::cusparseSpMatrix_DENSE_10010(const std::vector<float>& diag, memory_t mem_type)
    : spMatrix_DENSE<float>(diag, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE) {
    assert(mem_type != HOST);
    this->createDesc(CUDA_R_32F);
}
template <>
cusparseSpMatrix_DENSE_10010<double>::cusparseSpMatrix_DENSE_10010(const std::vector<double>& diag, memory_t mem_type)
    : spMatrix_DENSE<double>(diag, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE) {
    assert(mem_type != HOST);
    this->createDesc(CUDA_R_64F);
}
template <>
cusparseSpMatrix_DENSE_10010<int>::cusparseSpMatrix_DENSE_10010(unsigned dim0, unsigned dim1, memory_t mem_type)
    : spMatrix_DENSE<int>(dim0, dim1, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE) {
    assert(mem_type != HOST);
    this->createDesc(CUDA_R_32I);
}
template <>
cusparseSpMatrix_DENSE_10010<float>::cusparseSpMatrix_DENSE_10010(unsigned dim0, unsigned dim1, memory_t mem_type)
    : spMatrix_DENSE<float>(dim0, dim1, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE) {
    assert(mem_type != HOST);
    this->createDesc(CUDA_R_32F);
}
template <>
cusparseSpMatrix_DENSE_10010<double>::cusparseSpMatrix_DENSE_10010(unsigned dim0, unsigned dim1, memory_t mem_type)
    : spMatrix_DENSE<double>(dim0, dim1, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE) {
    assert(mem_type != HOST);
    this->createDesc(CUDA_R_64F);
}
//  explicit instantiation for type int, float, double
template class cusparseSpMatrix_DENSE_10010<int>;
template class cusparseSpMatrix_DENSE_10010<float>;
template class cusparseSpMatrix_DENSE_10010<double>;

//  for concrete class cusparseSparseMat
template <typename T>
void cusparseSpMatrix_CSR_10010<T>::createDesc(cudaDataType_t cuda_data_type) {
    _descriptor = new cusparseSpMatDescr_t;
    cusparseErrchk(cusparseCreateCsr(AS_TYPE(cusparseSpMatDescr_t*, _descriptor), _dim0, _dim1, _nnz,
                                     _rowCount->get_ptr(), _colIdx->get_ptr(), _valList->get_ptr(), CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_16U, CUSPARSE_INDEX_BASE_ZERO, cuda_data_type));
    _descriptor_is_set = true;
}
template <typename T>
cusparseSpMatrix_CSR_10010<T>::cusparseSpMatrix_CSR_10010(const Tensor<T>* spMatrixTensorPtr, memory_t mem_type)
    : spMatrix_CSR<T>(spMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR) {
    fprintf(stderr, "Requested template type for cusparseSpMatrix_CSR_10010 is not supported\n");
}
template <typename T>
cusparseSpMatrix_CSR_10010<T>::cusparseSpMatrix_CSR_10010(const std::vector<T>& diag, memory_t mem_type)
    : spMatrix_CSR<T>(diag, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR) {
    fprintf(stderr, "Requested template type for cusparseSpMatrix_CSR_10010 is not supported\n");
}
template <typename T>
cusparseSpMatrix_CSR_10010<T>::cusparseSpMatrix_CSR_10010(unsigned dim0, unsigned dim1, const std::vector<T>& valList,
                                                          const std::vector<int>& rowAccum,
                                                          const std::vector<int>& colIdx, memory_t mem_type)
    : spMatrix_CSR<T>(dim0, dim1, valList, rowAccum, colIdx, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR) {
    fprintf(stderr, "Requested template type for cusparseSpMatrix_CSR_10010 is not supported\n");
}
template <typename T>
cusparseSpMatrix_CSR_10010<T>::~cusparseSpMatrix_CSR_10010(void) {
    if (this->_descriptor_is_set) {
        cusparseDestroySpMat(*AS_TYPE(cusparseSpMatDescr_t*, this->_descriptor));
        this->_descriptor_is_set = false;
        //  todo: check if desctiptor is also freed (e.g. assigned nullptr)
    }
}
//  specialization
//  todo: simplify code
template <>
cusparseSpMatrix_CSR_10010<int>::cusparseSpMatrix_CSR_10010(const Tensor<int>* spMatrixTensorPtr, memory_t mem_type)
    : spMatrix_CSR<int>(spMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR) {
    assert(mem_type != HOST);
    this->createDesc(CUDA_R_32I);
}
template <>
cusparseSpMatrix_CSR_10010<float>::cusparseSpMatrix_CSR_10010(const Tensor<float>* spMatrixTensorPtr, memory_t mem_type)
    : spMatrix_CSR<float>(spMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR) {
    assert(mem_type != HOST);
    this->createDesc(CUDA_R_32F);
}
template <>
cusparseSpMatrix_CSR_10010<double>::cusparseSpMatrix_CSR_10010(const Tensor<double>* spMatrixTensorPtr,
                                                               memory_t mem_type)
    : spMatrix_CSR<double>(spMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR) {
    assert(mem_type != HOST);
    this->createDesc(CUDA_R_64F);
}
template <>
cusparseSpMatrix_CSR_10010<int>::cusparseSpMatrix_CSR_10010(const std::vector<int>& diag, memory_t mem_type)
    : spMatrix_CSR<int>(diag, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR) {
    assert(mem_type != HOST);
    this->createDesc(CUDA_R_32I);
}
template <>
cusparseSpMatrix_CSR_10010<float>::cusparseSpMatrix_CSR_10010(const std::vector<float>& diag, memory_t mem_type)
    : spMatrix_CSR<float>(diag, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR) {
    assert(mem_type != HOST);
    this->createDesc(CUDA_R_32F);
}
template <>
cusparseSpMatrix_CSR_10010<double>::cusparseSpMatrix_CSR_10010(const std::vector<double>& diag, memory_t mem_type)
    : spMatrix_CSR<double>(diag, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR) {
    assert(mem_type != HOST);
    this->createDesc(CUDA_R_64F);
}
template <>
cusparseSpMatrix_CSR_10010<int>::cusparseSpMatrix_CSR_10010(unsigned dim0, unsigned dim1,
                                                            const std::vector<int>& valList,
                                                            const std::vector<int>& rowAccum,
                                                            const std::vector<int>& colIdx, memory_t mem_type)
    : spMatrix_CSR<int>(dim0, dim1, valList, rowAccum, colIdx, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR) {
    assert(mem_type != HOST);
    this->createDesc(CUDA_R_32I);
}
template <>
cusparseSpMatrix_CSR_10010<float>::cusparseSpMatrix_CSR_10010(unsigned dim0, unsigned dim1,
                                                              const std::vector<float>& valList,
                                                              const std::vector<int>& rowAccum,
                                                              const std::vector<int>& colIdx, memory_t mem_type)
    : spMatrix_CSR<float>(dim0, dim1, valList, rowAccum, colIdx, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR) {
    assert(mem_type != HOST);
    this->createDesc(CUDA_R_32F);
}
template <>
cusparseSpMatrix_CSR_10010<double>::cusparseSpMatrix_CSR_10010(unsigned dim0, unsigned dim1,
                                                               const std::vector<double>& valList,
                                                               const std::vector<int>& rowAccum,
                                                               const std::vector<int>& colIdx, memory_t mem_type)
    : spMatrix_CSR<double>(dim0, dim1, valList, rowAccum, colIdx, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR) {
    assert(mem_type != HOST);
    this->createDesc(CUDA_R_64F);
}

//  explicit instantiation for type int, float, double
template class cusparseSpMatrix_CSR_10010<int>;
template class cusparseSpMatrix_CSR_10010<float>;
template class cusparseSpMatrix_CSR_10010<double>;
#endif
#endif
}  //  namespace spMatrix
}  //  namespace magmadnn
