#include "sparseMatrix/sparseMatrix.h"

namespace magmadnn {
namespace spMatrix {

//  for abstract class sparseMatrix
#if defined(DEBUG)
template <typename T>
sparseMatrix<T>::sparseMatrix(void) : _descriptor(nullptr), _descriptor_is_set(false), _dim0(0), _dim1(0) {
    fprintf(stderr, "Constructor for sparseMatrix called without parameters.\n");
}
#endif
template <typename T>
sparseMatrix<T>::sparseMatrix(const Tensor<T>* adjMatrixTensorPtr, memory_t mem_type, spMatrix_format format)
    : _format(format), _descriptor(nullptr), _descriptor_is_set(false), _mem_type(mem_type) {
    assert(T_IS_MATRIX(adjMatrixTensorPtr));
    _dim0 = adjMatrixTensorPtr->get_shape(0);
    _dim1 = adjMatrixTensorPtr->get_shape(1);
}
template <typename T>
sparseMatrix<T>::~sparseMatrix(void) {
    //  free _descriptor in corresponding class
    /* if (_descriptor_is_set) {
        delete _descriptor;
        _descriptor_is_set = false;
    } */
}

//  for abstract class spMatrix_DENSE
#if defined(DEBUG)
template <typename T>
spMatrix_DENSE<T>::spMatrix_DENSE(void) : sparseMatrix(void), _data(nullptr) {
    fprintf(stderr, "Constructor for spMatrix_DENSE called without parameters.\n");
}
#endif
template <typename T>
spMatrix_DENSE<T>::spMatrix_DENSE(const Tensor<T>* adjMatrixTensorPtr, memory_t mem_type, spMatrix_format format,
                                  bool copy)
    : sparseMatrix<T>(adjMatrixTensorPtr, mem_type, format) {
    _data = new Tensor<T>({this->_dim0, this->_dim1}, {NONE, {}}, mem_type);
    if (copy) {
        _data->copy_from(*adjMatrixTensorPtr);
    }
}
template <typename T>
spMatrix_DENSE<T>::~spMatrix_DENSE(void) {
    delete _data;
}

//  for abstract class spMatrix_CSR
#if defined(DEBUG)
template <typename T>
spMatrix_CSR<T>::spMatrix_CSR(void)
    : sparseMatrix(void), _nnz(0), _valList(nullptr), _rowCount(nullptr), _colIdx(nullptr) {
    fprintf(stderr, "Constructor for spMatrix_CSR called without parameters.\n");
}
#endif
template <typename T>
spMatrix_CSR<T>::spMatrix_CSR(const Tensor<T>* adjMatrixTensorPtr, memory_t mem_type, spMatrix_format format)
    : sparseMatrix<T>(adjMatrixTensorPtr, mem_type, format) {
    _nnz = 0;
    std::vector<T> nonzeroEleV;
    std::vector<int> rowAccV, colIdxV;
    rowAccV.push_back(0);
    unsigned rowCounter = 0;
    for (unsigned j = 0; j < this->_dim1; j++) {
        for (unsigned i = 0; i < this->_dim0; i++) {
            if (adjMatrixTensorPtr->get({i, j}) != T(0)) {
                nonzeroEleV.push_back(adjMatrixTensorPtr->get({i, j}));
                colIdxV.push_back(i);
                ++rowCounter;
            }
        }
        rowAccV.push_back(rowCounter);
    }
    _nnz = rowCounter;
    _valList = new Tensor<T>({unsigned(1), this->_nnz}, {NONE, {}}, mem_type);
    _rowCount = new Tensor<int>({unsigned(1), this->_dim0 + 1}, {NONE, {}}, mem_type);
    _colIdx = new Tensor<int>({unsigned(1), this->_nnz}, {NONE, {}}, mem_type);
    //  todo: better method?
    for (unsigned idx = 0; idx < this->_nnz; ++idx) {
        _valList->set(idx, nonzeroEleV[idx]);
        _colIdx->set(idx, colIdxV[idx]);
    }
    for (unsigned idx = 0; idx < this->_dim0 + 1; ++idx) {
        _rowCount->set(idx, rowAccV[idx]);
    }
}
template <typename T>
void spMatrix_CSR<T>::get_uncompressed_mat(Tensor<T>* output, T alpha) const {
    assert(T_IS_MATRIX(output));
    assert(output->get_shape(0) == this->_dim0);
    assert(output->get_shape(1) == this->_dim1);
    if (alpha == (T)0) {
        return;
    }
    unsigned nnzCount = 0;
    int count0, count1 = 0, toFill;
    for (unsigned rowIdx = 0; rowIdx < this->_dim0; ++rowIdx) {
        count0 = count1;
        count1 = _rowCount->get(rowIdx + 1);
        toFill = count1 - count0;
        for (unsigned colIdx = 0; colIdx < this->_dim1; ++colIdx) {
            if (toFill > 0 && colIdx == (unsigned) _colIdx->get(nnzCount)) {
                output->set({rowIdx, colIdx}, alpha * _valList->get(nnzCount++));
                toFill--;
            }
        }
    }
}
template <typename T>
spMatrix_CSR<T>::~spMatrix_CSR(void) {
    delete _valList;
    delete _rowCount;
    delete _colIdx;
}

//  for concrete class hostSpMatrix_DENSE
template <typename T>
hostSpMatrix_DENSE<T>::hostSpMatrix_DENSE(const Tensor<T>* adjMatrixTensorPtr, bool copy)
    : spMatrix_DENSE<T>(adjMatrixTensorPtr, HOST, SPARSEMATRIX_FORMAT_HOST_DENSE, copy) {
    /* empty */
}
template <typename T>
hostSpMatrix_DENSE<T>::~hostSpMatrix_DENSE(void) {
    /* empty */
}
//  explicit instantiation for type int, float, double
template class hostSpMatrix_DENSE<int>;
template class hostSpMatrix_DENSE<float>;
template class hostSpMatrix_DENSE<double>;

//  for concrete class hostSpMatrix_CSR
template <typename T>
hostSpMatrix_CSR<T>::hostSpMatrix_CSR(const Tensor<T>* adjMatrixTensorPtr)
    : spMatrix_CSR<T>(adjMatrixTensorPtr, HOST, SPARSEMATRIX_FORMAT_HOST_CSR) {
    /* empty */
}
template <typename T>
hostSpMatrix_CSR<T>::~hostSpMatrix_CSR(void) {
    /* empty */
}
//  explicit instantiation for type int, float, double
template class hostSpMatrix_CSR<int>;
template class hostSpMatrix_CSR<float>;
template class hostSpMatrix_CSR<double>;

#if defined(_HAS_CUDA_)

//  for concrete class cusparseSpMatrix_DENSE
template <typename T>
cusparseSpMatrix_DENSE<T>::cusparseSpMatrix_DENSE(const Tensor<T>* adjMatrixTensorPtr, memory_t mem_type, bool copy)
    : spMatrix_DENSE<T>(adjMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE, copy) {
    fprintf(stderr, "Requested template type for cusparseSpMatrix_DENSE is not supported\n");
}
template <typename T>
cusparseSpMatrix_DENSE<T>::~cusparseSpMatrix_DENSE(void) {
    if (this->_descriptor_is_set) {
        cusparseDestroyDnMat(*reinterpret_cast<cusparseDnMatDescr_t*>(this->_descriptor));
        this->_descriptor_is_set = false;
        //  todo: check if desctiptor is also freed (e.g. assigned nullptr)
    }
}
//  specialization
template <>
cusparseSpMatrix_DENSE<int>::cusparseSpMatrix_DENSE(const Tensor<int>* adjMatrixTensorPtr, memory_t mem_type, bool copy)
    : spMatrix_DENSE<int>(adjMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE, copy) {
    assert(mem_type != HOST);
    if (copy) {
        internal::transpose_full_device<int>(_data, _data);
    }
    _descriptor = new cusparseDnMatDescr_t;
    cusparseErrchk(cusparseCreateDnMat(reinterpret_cast<cusparseDnMatDescr_t*>(_descriptor), _dim0, _dim1, _dim0,
                                       _data->get_ptr(), CUDA_R_32I, CUSPARSE_ORDER_COL));
    _descriptor_is_set = true;
}
template <>
cusparseSpMatrix_DENSE<float>::cusparseSpMatrix_DENSE(const Tensor<float>* adjMatrixTensorPtr, memory_t mem_type,
                                                      bool copy)
    : spMatrix_DENSE<float>(adjMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE, copy) {
    assert(mem_type != HOST);
    if (copy) {
        internal::transpose_full_device<float>(_data, _data);
    }
    _descriptor = new cusparseDnMatDescr_t;
    cusparseErrchk(cusparseCreateDnMat(reinterpret_cast<cusparseDnMatDescr_t*>(_descriptor), _dim0, _dim1, _dim0,
                                       _data->get_ptr(), CUDA_R_32F, CUSPARSE_ORDER_COL));
    _descriptor_is_set = true;
}
template <>
cusparseSpMatrix_DENSE<double>::cusparseSpMatrix_DENSE(const Tensor<double>* adjMatrixTensorPtr, memory_t mem_type,
                                                       bool copy)
    : spMatrix_DENSE<double>(adjMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE, copy) {
    assert(mem_type != HOST);
    if (copy) {
        internal::transpose_full_device<double>(_data, _data);
    }
    _descriptor = new cusparseDnMatDescr_t;
    cusparseErrchk(cusparseCreateDnMat(reinterpret_cast<cusparseDnMatDescr_t*>(_descriptor), _dim0, _dim1, _dim0,
                                       _data->get_ptr(), CUDA_R_64F, CUSPARSE_ORDER_COL));
    _descriptor_is_set = true;
}
//  explicit instantiation for type int, float, double
template class cusparseSpMatrix_DENSE<int>;
template class cusparseSpMatrix_DENSE<float>;
template class cusparseSpMatrix_DENSE<double>;

//  for concrete class cusparseSparseMat
template <typename T>
cusparseSpMatrix_CSR<T>::cusparseSpMatrix_CSR(const Tensor<T>* adjMatrixTensorPtr, memory_t mem_type)
    : spMatrix_CSR<T>(adjMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR) {
    fprintf(stderr, "Requested template type for cusparseSpMatrix_CSR is not supported\n");
}
template <typename T>
cusparseSpMatrix_CSR<T>::~cusparseSpMatrix_CSR(void) {
    if (this->_descriptor_is_set) {
        cusparseDestroySpMat(*reinterpret_cast<cusparseSpMatDescr_t*>(this->_descriptor));
        this->_descriptor_is_set = false;
        //  todo: check if desctiptor is also freed (e.g. assigned nullptr)
    }
}
//  specialization
template <>
cusparseSpMatrix_CSR<int>::cusparseSpMatrix_CSR(const Tensor<int>* adjMatrixTensorPtr, memory_t mem_type)
    : spMatrix_CSR<int>(adjMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR) {
    assert(mem_type != HOST);
    _descriptor = new cusparseSpMatDescr_t;
    cusparseErrchk(cusparseCreateCsr(reinterpret_cast<cusparseSpMatDescr_t*>(_descriptor), _dim0, _dim1, _nnz,
                                     _rowCount->get_ptr(), _colIdx->get_ptr(), _valList->get_ptr(), CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_16U, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32I));
    _descriptor_is_set = true;
}
template <>
cusparseSpMatrix_CSR<float>::cusparseSpMatrix_CSR(const Tensor<float>* adjMatrixTensorPtr, memory_t mem_type)
    : spMatrix_CSR<float>(adjMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR) {
    assert(mem_type != HOST);
    _descriptor = new cusparseSpMatDescr_t;
    cusparseErrchk(cusparseCreateCsr(reinterpret_cast<cusparseSpMatDescr_t*>(_descriptor), _dim0, _dim1, _nnz,
                                     _rowCount->get_ptr(), _colIdx->get_ptr(), _valList->get_ptr(), CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_16U, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    _descriptor_is_set = true;
}
template <>
cusparseSpMatrix_CSR<double>::cusparseSpMatrix_CSR(const Tensor<double>* adjMatrixTensorPtr, memory_t mem_type)
    : spMatrix_CSR<double>(adjMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR) {
    assert(mem_type != HOST);
    _descriptor = new cusparseSpMatDescr_t;
    cusparseErrchk(cusparseCreateCsr(reinterpret_cast<cusparseSpMatDescr_t*>(_descriptor), _dim0, _dim1, _nnz,
                                     _rowCount->get_ptr(), _colIdx->get_ptr(), _valList->get_ptr(), CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_16U, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    _descriptor_is_set = true;
}
//  explicit instantiation for type int, float, double
template class cusparseSpMatrix_CSR<int>;
template class cusparseSpMatrix_CSR<float>;
template class cusparseSpMatrix_CSR<double>;

#endif
}  // namespace spMatrix
}  // namespace magmadnn
