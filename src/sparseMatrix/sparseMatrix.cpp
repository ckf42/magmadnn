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
sparseMatrix<T>::sparseMatrix(const Tensor<T>* spMatrixTensorPtr, memory_t mem_type, spMatrix_format format)
    : _format(format), _descriptor(nullptr), _descriptor_is_set(false), _mem_type(mem_type) {
    assert(T_IS_MATRIX(spMatrixTensorPtr));
    _dim0 = spMatrixTensorPtr->get_shape(0);
    _dim1 = spMatrixTensorPtr->get_shape(1);
}
template <typename T>
sparseMatrix<T>::sparseMatrix(unsigned dim0, unsigned dim1, memory_t mem_type, spMatrix_format format)
    : _format(format), _dim0(dim0), _dim1(dim1), _descriptor(nullptr), _descriptor_is_set(false), _mem_type(mem_type) {
    /* empty */
}
template <typename T>
sparseMatrix<T>::~sparseMatrix(void) {
    //  free _descriptor in corresponding class
    /* if (_descriptor_is_set) {
        delete _descriptor;
        _descriptor_is_set = false;
    } */
}
template <typename T>
T sparseMatrix<T>::rowSum(unsigned rowIdx) const {  //  default rowSum, if no override
    assert(rowIdx < this->_dim0);
    T out = (T) 0;
    for (auto& w : this->getRowSlice(rowIdx)) {
        out += w;
    }
    return out;
}

//  for abstract class spMatrix_DENSE
#if defined(DEBUG)
template <typename T>
spMatrix_DENSE<T>::spMatrix_DENSE(void) : sparseMatrix(void), _data(nullptr) {
    fprintf(stderr, "Constructor for spMatrix_DENSE called without parameters.\n");
}
#endif
template <typename T>
spMatrix_DENSE<T>::spMatrix_DENSE(const Tensor<T>* spMatrixTensorPtr, memory_t mem_type, spMatrix_format format,
                                  bool copy)
    : sparseMatrix<T>(spMatrixTensorPtr, mem_type, format) {
    _data = new Tensor<T>({this->_dim0, this->_dim1}, {NONE, {}}, mem_type);
    if (copy) {
        _data->copy_from(*spMatrixTensorPtr);
    }
}
template <typename T>
spMatrix_DENSE<T>::spMatrix_DENSE(const std::vector<T>& diag, memory_t mem_type, spMatrix_format format)
    : sparseMatrix<T>(diag.size(), diag.size(), mem_type, format) {
    assert(this->_dim0 != 0);
    assert(this->_dim1 != 0);
    _data = new Tensor<T>({this->_dim0, this->_dim1}, {ZERO, {}}, mem_type);
    for (unsigned idx = 0; idx < this->_dim0; ++idx) {
        if (diag[idx] != (T) 0) {
            _data->set({idx, idx}, diag[idx]);
        }
    }
}
template <typename T>
spMatrix_DENSE<T>::spMatrix_DENSE(unsigned dim0, unsigned dim1, memory_t mem_type, spMatrix_format format)
    : sparseMatrix<T>(dim0, dim1, mem_type, format) {
    _data = new Tensor<T>({this->_dim0, this->_dim1}, {ZERO, {}}, mem_type);
}
template <typename T>
spMatrix_DENSE<T>::~spMatrix_DENSE(void) {
    delete _data;
}
template <typename T>
std::vector<T> spMatrix_DENSE<T>::getRowSlice(unsigned rowIdx) const {
    assert(rowIdx < this->_dim0);
    std::vector<T> out(this->_dim1);
    for (unsigned idx = 0; idx < this->_dim1; ++idx) {
        out[idx] = this->get(rowIdx, idx);
    }
    return out;
}
template <typename T>
T spMatrix_DENSE<T>::rowSum(unsigned rowIdx) const {
    assert(rowIdx < this->_dim1);
    T out = (T) 0;
    for (unsigned idx = 0; idx < this->_dim1; ++idx) {
        out += this->get(rowIdx, idx);
    }
    return out;
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
spMatrix_CSR<T>::spMatrix_CSR(const Tensor<T>* spMatrixTensorPtr, memory_t mem_type, spMatrix_format format)
    : sparseMatrix<T>(spMatrixTensorPtr, mem_type, format) {
    _nnz = 0;
    std::vector<T> nonzeroEleV;
    std::vector<int> rowAccV, colIdxV;
    rowAccV.push_back(0);
    unsigned rowCounter = 0;
    for (unsigned i = 0; i < this->_dim0; i++) {
        for (unsigned j = 0; j < this->_dim1; j++) {
            if (spMatrixTensorPtr->get({i, j}) != T(0)) {
                nonzeroEleV.push_back(spMatrixTensorPtr->get({i, j}));
                colIdxV.push_back(j);
                ++rowCounter;
            }
        }
        rowAccV.push_back(rowCounter);
    }
    _nnz = rowCounter;
    _valList = new Tensor<T>({unsigned(1), this->_nnz}, {NONE, {}}, mem_type);
    _rowCount = new Tensor<int>({unsigned(1), this->_dim0 + 1}, {NONE, {}}, mem_type);
    _colIdx = new Tensor<int>({unsigned(1), this->_nnz}, {NONE, {}}, mem_type);
    //  todo: batch assign?
    for (unsigned idx = 0; idx < this->_nnz; ++idx) {
        _valList->set(idx, nonzeroEleV[idx]);
        _colIdx->set(idx, colIdxV[idx]);
    }
    for (unsigned idx = 0; idx < this->_dim0 + 1; ++idx) {
        _rowCount->set(idx, rowAccV[idx]);
    }
}
template <typename T>
spMatrix_CSR<T>::spMatrix_CSR(const std::vector<T>& diag, memory_t mem_type, spMatrix_format format)
    : sparseMatrix<T>(diag.size(), diag.size(), mem_type, format) {
    _nnz = 0;
    std::vector<T> valV;
    std::vector<int> rowV, colV;
    rowV.push_back(0);
    for (unsigned idx = 0; idx < this->_dim0; ++idx) {
        if (diag[idx] != (T) 0) {
            _nnz++;
            valV.push_back(diag[idx]);
            colV.push_back(idx);
        }
        rowV.push_back(_nnz);
    }
    _valList = new Tensor<T>({(unsigned) 1, _nnz}, {NONE, {}}, mem_type);
    _rowCount = new Tensor<int>({(unsigned) 1, this->_dim0 + 1}, {NONE, {}}, mem_type);
    _colIdx = new Tensor<int>({(unsigned) 1, _nnz}, {NONE, {}}, mem_type);
    for (unsigned idx = 0; idx < this->_nnz; ++idx) {
        _valList->set(idx, valV[idx]);
        _colIdx->set(idx, colV[idx]);
    }
    for (unsigned idx = 0; idx < this->_dim0 + 1; ++idx) {
        _rowCount->set(idx, rowV[idx]);
    }
}
template <typename T>
spMatrix_CSR<T>::spMatrix_CSR(unsigned dim0, unsigned dim1, const std::vector<T>& valList,
                              const std::vector<int>& rowAccum, const std::vector<int>& colIdx, memory_t mem_type,
                              spMatrix_format format)
    : sparseMatrix<T>(dim0, dim1, mem_type, format), _nnz(valList.size()) {
    assert(this->_nnz == colIdx.size());
    assert(dim0 + 1 == rowAccum.size());
    _valList = new Tensor<T>({(unsigned) 1, _nnz}, {NONE, {}}, mem_type);
    _rowCount = new Tensor<int>({(unsigned) 1, dim0 + 1}, {NONE, {}}, mem_type);
    _colIdx = new Tensor<int>({(unsigned) 1, _nnz}, {NONE, {}}, mem_type);
    for (unsigned idx = 0; idx < _nnz; ++idx) {
        _valList->set(idx, valList[idx]);
        _colIdx->set(idx, colIdx[idx]);
    }
    for (unsigned idx = 0; idx < this->_dim0 + 1; ++idx) {
        _rowCount->set(idx, rowAccum[idx]);
    }
}
template <typename T>
void spMatrix_CSR<T>::get_uncompressed_mat(Tensor<T>* output, T alpha) const {
    assert(T_IS_MATRIX(output));
    assert(output->get_shape(0) == this->_dim0);
    assert(output->get_shape(1) == this->_dim1);
    if (alpha == (T) 0) {
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
template <typename T>
T spMatrix_CSR<T>::get(unsigned i, unsigned j) const {
    assert(i < this->_dim0);
    assert(j < this->_dim1);
    int lastRowAcc = _rowCount->get(i), thisRowAcc = _rowCount->get(i + 1);
    if (lastRowAcc == thisRowAcc) {  //  row i is all zero
        return (T) 0;
    }
    //  nonzero idx of row i is in _colIdx[lastRowAcc:thisRowAcc)
    //  use lastRowAcc as an idx
    //  todo: change to use binary search?
    while (lastRowAcc != thisRowAcc && (unsigned) _colIdx->get(lastRowAcc) < j) {
        ++lastRowAcc;
    }
    if (lastRowAcc != thisRowAcc && (unsigned) _colIdx->get(lastRowAcc) == j) {
        return _valList->get(lastRowAcc);
    }
    return (T) 0;
}
template <typename T>
std::vector<T> spMatrix_CSR<T>::getRowSlice(unsigned rowIdx) const {
    assert(rowIdx < this->_dim0);
    int lastRowAcc = _rowCount->get(rowIdx), thisRowAcc = _rowCount->get(rowIdx + 1);
    std::vector<T> out(this->_dim1);
    for (unsigned idx = 0; idx < this->_dim1; ++idx) {
        while (lastRowAcc < thisRowAcc && (unsigned) _colIdx->get(lastRowAcc) < idx) {
            ++lastRowAcc;
        }
        if (lastRowAcc == thisRowAcc || (unsigned) _colIdx->get(lastRowAcc) != idx) {
            out[idx] = (T) 0;
        } else {
            out[idx] = _valList->get(lastRowAcc);
        }
    }
    return out;
}
template <typename T>
void spMatrix_CSR<T>::set(unsigned i, unsigned j, T val) {
    assert(i < this->_dim0);
    assert(j < this->_dim1);
    T currVal = (T) 0;
    int idx = _rowCount->get(i), thisRowAcc = _rowCount->get(i + 1);
    while (idx != thisRowAcc) {
        if ((unsigned) _colIdx->get(idx) == j) {
            currVal = _valList->get(idx);
            break;
        }
        ++idx;
    }
    if (currVal == val) {
        return;
    }
    //  don't know how to easily change _valList, _colIdx, _descriptor yet
    //  todo: implement this
    assert(currVal != (T) 0 && val != (T) 0);
    _valList->set(idx, val);
}
template <typename T>
T spMatrix_CSR<T>::rowSum(unsigned rowIdx) const {
    assert(rowIdx < this->_dim0);
    T out = (T) 0;
    for (int iter = _rowCount->get(rowIdx); iter != _rowCount->get(rowIdx + 1); ++iter) {
        out += _valList->get(iter);
    }
    return out;
}

//  for concrete class hostSpMatrix_DENSE
template <typename T>
hostSpMatrix_DENSE<T>::hostSpMatrix_DENSE(const Tensor<T>* spMatrixTensorPtr, bool copy)
    : spMatrix_DENSE<T>(spMatrixTensorPtr, HOST, SPARSEMATRIX_FORMAT_HOST_DENSE, copy) {
    /* empty */
}
template <typename T>
hostSpMatrix_DENSE<T>::hostSpMatrix_DENSE(const std::vector<T>& diag)
    : spMatrix_DENSE<T>(diag, HOST, SPARSEMATRIX_FORMAT_HOST_DENSE) {
    /* empty */
}
template <typename T>
hostSpMatrix_DENSE<T>::hostSpMatrix_DENSE(unsigned dim0, unsigned dim1)
    : spMatrix_DENSE<T>(dim0, dim1, HOST, SPARSEMATRIX_FORMAT_HOST_DENSE) {
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
hostSpMatrix_CSR<T>::hostSpMatrix_CSR(const Tensor<T>* spMatrixTensorPtr)
    : spMatrix_CSR<T>(spMatrixTensorPtr, HOST, SPARSEMATRIX_FORMAT_HOST_CSR) {
    /* empty */
}
template <typename T>
hostSpMatrix_CSR<T>::hostSpMatrix_CSR(const std::vector<T>& diag)
    : spMatrix_CSR<T>(diag, HOST, SPARSEMATRIX_FORMAT_HOST_CSR) {
    /* empty */
}
template <typename T>
hostSpMatrix_CSR<T>::hostSpMatrix_CSR(unsigned dim0, unsigned dim1, const std::vector<T>& valList,
                                      const std::vector<int>& rowAccum, const std::vector<int>& colIdx)
    : spMatrix_CSR<T>(dim0, dim1, valList, rowAccum, colIdx, HOST, SPARSEMATRIX_FORMAT_HOST_CSR) {
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
#if (CUDART_VERSION >= 100100)
//  todo: clean up
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
void cusparseSpMatrix_DENSE_10010<T>::createDesc(void) {
    this->_descriptor = new cusparseDnMatDescr_t;
    cusparseErrchk(cusparseCreateDnMat(AS_TYPE(cusparseDnMatDescr_t*, this->_descriptor), this->_dim1, this->_dim0,
                                       this->_dim1, this->_data->get_ptr(), this->cuda_data_type, CUSPARSE_ORDER_COL));
    this->_descriptor_is_set = true;
}
//  specialization
template <>
cusparseSpMatrix_DENSE_10010<int>::cusparseSpMatrix_DENSE_10010(const Tensor<int>* spMatrixTensorPtr, memory_t mem_type,
                                                                bool copy)
    : spMatrix_DENSE<int>(spMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE, copy),
      cuda_data_type(CUDA_R_32I) {
    assert(mem_type != HOST);
    this->createDesc();
}
template <>
cusparseSpMatrix_DENSE_10010<float>::cusparseSpMatrix_DENSE_10010(const Tensor<float>* spMatrixTensorPtr,
                                                                  memory_t mem_type, bool copy)
    : spMatrix_DENSE<float>(spMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE, copy),
      cuda_data_type(CUDA_R_32F) {
    assert(mem_type != HOST);
    this->createDesc();
}
template <>
cusparseSpMatrix_DENSE_10010<double>::cusparseSpMatrix_DENSE_10010(const Tensor<double>* spMatrixTensorPtr,
                                                                   memory_t mem_type, bool copy)
    : spMatrix_DENSE<double>(spMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE, copy),
      cuda_data_type(CUDA_R_64F) {
    assert(mem_type != HOST);
    this->createDesc();
}
template <>
cusparseSpMatrix_DENSE_10010<int>::cusparseSpMatrix_DENSE_10010(const std::vector<int>& diag, memory_t mem_type)
    : spMatrix_DENSE<int>(diag, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE), cuda_data_type(CUDA_R_32I) {
    assert(mem_type != HOST);
    this->createDesc();
}
template <>
cusparseSpMatrix_DENSE_10010<float>::cusparseSpMatrix_DENSE_10010(const std::vector<float>& diag, memory_t mem_type)
    : spMatrix_DENSE<float>(diag, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE), cuda_data_type(CUDA_R_32F) {
    assert(mem_type != HOST);
    this->createDesc();
}
template <>
cusparseSpMatrix_DENSE_10010<double>::cusparseSpMatrix_DENSE_10010(const std::vector<double>& diag, memory_t mem_type)
    : spMatrix_DENSE<double>(diag, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE), cuda_data_type(CUDA_R_64F) {
    assert(mem_type != HOST);
    this->createDesc();
}
template <>
cusparseSpMatrix_DENSE_10010<int>::cusparseSpMatrix_DENSE_10010(unsigned dim0, unsigned dim1, memory_t mem_type)
    : spMatrix_DENSE<int>(dim0, dim1, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE), cuda_data_type(CUDA_R_32I) {
    assert(mem_type != HOST);
    this->createDesc();
}
template <>
cusparseSpMatrix_DENSE_10010<float>::cusparseSpMatrix_DENSE_10010(unsigned dim0, unsigned dim1, memory_t mem_type)
    : spMatrix_DENSE<float>(dim0, dim1, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE), cuda_data_type(CUDA_R_32F) {
    assert(mem_type != HOST);
    this->createDesc();
}
template <>
cusparseSpMatrix_DENSE_10010<double>::cusparseSpMatrix_DENSE_10010(unsigned dim0, unsigned dim1, memory_t mem_type)
    : spMatrix_DENSE<double>(dim0, dim1, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE), cuda_data_type(CUDA_R_64F) {
    assert(mem_type != HOST);
    this->createDesc();
}
//  explicit instantiation for type int, float, double
template class cusparseSpMatrix_DENSE_10010<int>;
template class cusparseSpMatrix_DENSE_10010<float>;
template class cusparseSpMatrix_DENSE_10010<double>;

//  for concrete class cusparseSparseMat
template <typename T>
void cusparseSpMatrix_CSR_10010<T>::createDesc(void) {
    this->_descriptor = new cusparseSpMatDescr_t;
    cusparseErrchk(cusparseCreateCsr(AS_TYPE(cusparseSpMatDescr_t*, this->_descriptor), this->_dim0, this->_dim1,
                                     this->_nnz, this->_rowCount->get_ptr(), this->_colIdx->get_ptr(),
                                     this->_valList->get_ptr(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_16U,
                                     CUSPARSE_INDEX_BASE_ZERO, this->cuda_data_type));
    this->_descriptor_is_set = true;
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
template <>
cusparseSpMatrix_CSR_10010<int>::cusparseSpMatrix_CSR_10010(const Tensor<int>* spMatrixTensorPtr, memory_t mem_type)
    : spMatrix_CSR<int>(spMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR), cuda_data_type(CUDA_R_32I) {
    assert(mem_type != HOST);
    this->createDesc();
}
template <>
cusparseSpMatrix_CSR_10010<float>::cusparseSpMatrix_CSR_10010(const Tensor<float>* spMatrixTensorPtr, memory_t mem_type)
    : spMatrix_CSR<float>(spMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR), cuda_data_type(CUDA_R_32F) {
    assert(mem_type != HOST);
    this->createDesc();
}
template <>
cusparseSpMatrix_CSR_10010<double>::cusparseSpMatrix_CSR_10010(const Tensor<double>* spMatrixTensorPtr,
                                                               memory_t mem_type)
    : spMatrix_CSR<double>(spMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR), cuda_data_type(CUDA_R_64F) {
    assert(mem_type != HOST);
    this->createDesc();
}
template <>
cusparseSpMatrix_CSR_10010<int>::cusparseSpMatrix_CSR_10010(const std::vector<int>& diag, memory_t mem_type)
    : spMatrix_CSR<int>(diag, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR), cuda_data_type(CUDA_R_32I) {
    assert(mem_type != HOST);
    this->createDesc();
}
template <>
cusparseSpMatrix_CSR_10010<float>::cusparseSpMatrix_CSR_10010(const std::vector<float>& diag, memory_t mem_type)
    : spMatrix_CSR<float>(diag, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR), cuda_data_type(CUDA_R_32F) {
    assert(mem_type != HOST);
    this->createDesc();
}
template <>
cusparseSpMatrix_CSR_10010<double>::cusparseSpMatrix_CSR_10010(const std::vector<double>& diag, memory_t mem_type)
    : spMatrix_CSR<double>(diag, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR), cuda_data_type(CUDA_R_64F) {
    assert(mem_type != HOST);
    this->createDesc();
}
template <>
cusparseSpMatrix_CSR_10010<int>::cusparseSpMatrix_CSR_10010(unsigned dim0, unsigned dim1,
                                                            const std::vector<int>& valList,
                                                            const std::vector<int>& rowAccum,
                                                            const std::vector<int>& colIdx, memory_t mem_type)
    : spMatrix_CSR<int>(dim0, dim1, valList, rowAccum, colIdx, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR),
      cuda_data_type(CUDA_R_32I) {
    assert(mem_type != HOST);
    this->createDesc();
}
template <>
cusparseSpMatrix_CSR_10010<float>::cusparseSpMatrix_CSR_10010(unsigned dim0, unsigned dim1,
                                                              const std::vector<float>& valList,
                                                              const std::vector<int>& rowAccum,
                                                              const std::vector<int>& colIdx, memory_t mem_type)
    : spMatrix_CSR<float>(dim0, dim1, valList, rowAccum, colIdx, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR),
      cuda_data_type(CUDA_R_32F) {
    assert(mem_type != HOST);
    this->createDesc();
}
template <>
cusparseSpMatrix_CSR_10010<double>::cusparseSpMatrix_CSR_10010(unsigned dim0, unsigned dim1,
                                                               const std::vector<double>& valList,
                                                               const std::vector<int>& rowAccum,
                                                               const std::vector<int>& colIdx, memory_t mem_type)
    : spMatrix_CSR<double>(dim0, dim1, valList, rowAccum, colIdx, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR),
      cuda_data_type(CUDA_R_64F) {
    assert(mem_type != HOST);
    this->createDesc();
}

//  explicit instantiation for type int, float, double
template class cusparseSpMatrix_CSR_10010<int>;
template class cusparseSpMatrix_CSR_10010<float>;
template class cusparseSpMatrix_CSR_10010<double>;
#elif (CUDART_VERSION < 100100)
template <typename T>
cusparseSpMatrix_DENSE_LEGACY<T>::cusparseSpMatrix_DENSE_LEGACY(const Tensor<T>* spMatrixTenorPtr, memory_t mem_type,
                                                                bool copy)
    : spMatrix_DENSE<T>(0, 0, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE), cuda_data_type(CUDA_R_32I) {
    assert(mem_type != HOST);
    std::fprintf(stderr, "Requested template type for cusparseSpMatrix_DENSE_LEGACY is not supported\n");
}
template <>
cusparseSpMatrix_DENSE_LEGACY<float>::cusparseSpMatrix_DENSE_LEGACY(const Tensor<float>* spMatrixTenorPtr,
                                                                    memory_t mem_type, bool copy)
    : spMatrix_DENSE<float>(spMatrixTenorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE, copy),
      cuda_data_type(CUDA_R_32F) {
    assert(mem_type != HOST);
}
template <>
cusparseSpMatrix_DENSE_LEGACY<double>::cusparseSpMatrix_DENSE_LEGACY(const Tensor<double>* spMatrixTenorPtr,
                                                                     memory_t mem_type, bool copy)
    : spMatrix_DENSE<double>(spMatrixTenorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE, copy),
      cuda_data_type(CUDA_R_64F) {
    assert(mem_type != HOST);
}
template <typename T>
cusparseSpMatrix_DENSE_LEGACY<T>::cusparseSpMatrix_DENSE_LEGACY(const std::vector<T>& diag, memory_t mem_type)
    : spMatrix_DENSE<T>(0, 0, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE), cuda_data_type(CUDA_R_32I) {
    assert(mem_type != HOST);
    std::fprintf(stderr, "Requested template type for cusparseSpMatrix_DENSE_LEGACY is not supported\n");
}
template <>
cusparseSpMatrix_DENSE_LEGACY<float>::cusparseSpMatrix_DENSE_LEGACY(const std::vector<float>& diag, memory_t mem_type)
    : spMatrix_DENSE<float>(diag, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE), cuda_data_type(CUDA_R_32F) {
    assert(mem_type != HOST);
}
template <>
cusparseSpMatrix_DENSE_LEGACY<double>::cusparseSpMatrix_DENSE_LEGACY(const std::vector<double>& diag, memory_t mem_type)
    : spMatrix_DENSE<double>(diag, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE), cuda_data_type(CUDA_R_64F) {
    assert(mem_type != HOST);
}
template <typename T>
cusparseSpMatrix_DENSE_LEGACY<T>::cusparseSpMatrix_DENSE_LEGACY(unsigned dim0, unsigned dim1, memory_t mem_type)
    : spMatrix_DENSE<T>(0, 0, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE), cuda_data_type(CUDA_R_32I) {
    assert(mem_type != HOST);
    std::fprintf(stderr, "Requested template type for cusparseSpMatrix_DENSE_LEGACY is not supported\n");
}
template <>
cusparseSpMatrix_DENSE_LEGACY<float>::cusparseSpMatrix_DENSE_LEGACY(unsigned dim0, unsigned dim1, memory_t mem_type)
    : spMatrix_DENSE<float>(dim0, dim1, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE), cuda_data_type(CUDA_R_32F) {
    assert(mem_type != HOST);
}
template <>
cusparseSpMatrix_DENSE_LEGACY<double>::cusparseSpMatrix_DENSE_LEGACY(unsigned dim0, unsigned dim1, memory_t mem_type)
    : spMatrix_DENSE<double>(dim0, dim1, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE), cuda_data_type(CUDA_R_64F) {
    assert(mem_type != HOST);
}
template <typename T>
cusparseSpMatrix_DENSE_LEGACY<T>::~cusparseSpMatrix_DENSE_LEGACY(void) {
    /* empty */
}
template class cusparseSpMatrix_DENSE_LEGACY<int>;  //  no int
template class cusparseSpMatrix_DENSE_LEGACY<float>;
template class cusparseSpMatrix_DENSE_LEGACY<double>;

template <typename T>
void cusparseSpMatrix_CSR_LEGACY<T>::createDesc(void) {
    this->_descriptor = new cusparseMatDescr_t;
    cusparseErrchk(cusparseCreateMatDescr(AS_TYPE(cusparseMatDescr_t*, this->_descriptor)));
    cusparseErrchk(cusparseSetMatIndexBase(*AS_TYPE(cusparseMatDescr_t*, this->_descriptor), CUSPARSE_INDEX_BASE_ZERO));
    cusparseErrchk(cusparseSetMatType(*AS_TYPE(cusparseMatDescr_t*, this->_descriptor), CUSPARSE_MATRIX_TYPE_GENERAL));
    this->_descriptor_is_set = true;
}
template <typename T>
cusparseSpMatrix_CSR_LEGACY<T>::cusparseSpMatrix_CSR_LEGACY(const Tensor<T>* spMatrixTensorPtr, memory_t mem_type)
    : spMatrix_CSR<T>(spMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR), cuda_data_type(CUDA_R_32I) {
    assert(mem_type != HOST);
    std::fprintf(stderr, "Requested template type for cusparseSpMatrix_CSR_LEGACY is not supported\n");
}
template <>
cusparseSpMatrix_CSR_LEGACY<float>::cusparseSpMatrix_CSR_LEGACY(const Tensor<float>* spMatrixTensorPtr,
                                                                memory_t mem_type)
    : spMatrix_CSR<float>(spMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR), cuda_data_type(CUDA_R_32F) {
    assert(mem_type != HOST);
    this->createDesc();
}
template <>
cusparseSpMatrix_CSR_LEGACY<double>::cusparseSpMatrix_CSR_LEGACY(const Tensor<double>* spMatrixTensorPtr,
                                                                 memory_t mem_type)
    : spMatrix_CSR<double>(spMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR), cuda_data_type(CUDA_R_64F) {
    assert(mem_type != HOST);
    this->createDesc();
}
template <typename T>
cusparseSpMatrix_CSR_LEGACY<T>::cusparseSpMatrix_CSR_LEGACY(const std::vector<T>& diag, memory_t mem_type)
    : spMatrix_CSR<T>(diag, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR), cuda_data_type(CUDA_R_32I) {
    assert(mem_type != HOST);
    std::fprintf(stderr, "Requested template type for cusparseSpMatrix_CSR_LEGACY is not supported\n");
}
template <>
cusparseSpMatrix_CSR_LEGACY<float>::cusparseSpMatrix_CSR_LEGACY(const std::vector<float>& diag, memory_t mem_type)
    : spMatrix_CSR<float>(diag, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR), cuda_data_type(CUDA_R_32F) {
    assert(mem_type != HOST);
    this->createDesc();
}
template <>
cusparseSpMatrix_CSR_LEGACY<double>::cusparseSpMatrix_CSR_LEGACY(const std::vector<double>& diag, memory_t mem_type)
    : spMatrix_CSR<double>(diag, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR), cuda_data_type(CUDA_R_64F) {
    assert(mem_type != HOST);
    this->createDesc();
}
template <typename T>
cusparseSpMatrix_CSR_LEGACY<T>::cusparseSpMatrix_CSR_LEGACY(unsigned dim0, unsigned dim1, const std::vector<T>& valList,
                                                            const std::vector<int>& rowAccum,
                                                            const std::vector<int>& colIdx, memory_t mem_type)
    : spMatrix_CSR<T>(dim0, dim1, valList, rowAccum, colIdx, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR),
      cuda_data_type(CUDA_R_32I) {
    assert(mem_type != HOST);
    std::fprintf(stderr, "Requested template type for cusparseSpMatrix_CSR_LEGACY is not supported\n");
}
template <>
cusparseSpMatrix_CSR_LEGACY<float>::cusparseSpMatrix_CSR_LEGACY(unsigned dim0, unsigned dim1,
                                                                const std::vector<float>& valList,
                                                                const std::vector<int>& rowAccum,
                                                                const std::vector<int>& colIdx, memory_t mem_type)
    : spMatrix_CSR<float>(dim0, dim1, valList, rowAccum, colIdx, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR),
      cuda_data_type(CUDA_R_32F) {
    assert(mem_type != HOST);
    this->createDesc();
}
template <>
cusparseSpMatrix_CSR_LEGACY<double>::cusparseSpMatrix_CSR_LEGACY(unsigned dim0, unsigned dim1,
                                                                 const std::vector<double>& valList,
                                                                 const std::vector<int>& rowAccum,
                                                                 const std::vector<int>& colIdx, memory_t mem_type)
    : spMatrix_CSR<double>(dim0, dim1, valList, rowAccum, colIdx, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR),
      cuda_data_type(CUDA_R_64F) {
    assert(mem_type != HOST);
    this->createDesc();
}
template <typename T>
cusparseSpMatrix_CSR_LEGACY<T>::~cusparseSpMatrix_CSR_LEGACY(void) {
    if (this->_descriptor_is_set) {
        cusparseDestroyMatDescr(*AS_TYPE(cusparseMatDescr_t*, this->_descriptor));
        delete AS_TYPE(cusparseMatDescr_t*, this->_descriptor);
        this->_descriptor_is_set = false;
    }
}
template class cusparseSpMatrix_CSR_LEGACY<int>;
template class cusparseSpMatrix_CSR_LEGACY<float>;
template class cusparseSpMatrix_CSR_LEGACY<double>;
#endif
#endif

//  todo: better way to do this (with template)?
template <typename T>
sparseMatrix<T>* get_spMat(const Tensor<T>* spMatrixTensorPtr, spMatrix_format format, memory_t mem_type) {
    sparseMatrix<T>* out = nullptr;
    switch (format) {
        case SPARSEMATRIX_FORMAT_HOST_CSR:
            assert(mem_type == HOST);
            out = new hostSpMatrix_CSR<T>(spMatrixTensorPtr);
            break;
        case SPARSEMATRIX_FORMAT_HOST_DENSE:
            assert(mem_type == HOST);
            out = new hostSpMatrix_DENSE<T>(spMatrixTensorPtr);
            break;
#if defined(_HAS_CUDA_)
        case SPARSEMATRIX_FORMAT_CUSPARSE_DENSE:
            assert(mem_type != HOST);
            out = new cusparseSpMatrix_DENSE<T>(spMatrixTensorPtr, mem_type);
            break;
        case SPARSEMATRIX_FORMAT_CUSPARSE_CSR:
            assert(mem_type != HOST);
            out = new cusparseSpMatrix_CSR<T>(spMatrixTensorPtr, mem_type);
            break;
#endif
        default:
            std::fprintf(stderr, "unable to create a sparse matrix with unknown format.\n");
            break;
    }
    return out;
}
template sparseMatrix<int>* get_spMat(const Tensor<int>* spMatrixTensorPtr, spMatrix_format format, memory_t mem_type);
template sparseMatrix<float>* get_spMat(const Tensor<float>* spMatrixTensorPtr, spMatrix_format format,
                                        memory_t mem_type);
template sparseMatrix<double>* get_spMat(const Tensor<double>* spMatrixTensorPtr, spMatrix_format format,
                                         memory_t mem_type);

template <typename T>
sparseMatrix<T>* get_spMat(unsigned dim0, unsigned dim1, const std::vector<T>& valList,
                           const std::vector<int>& rowAccum, const std::vector<int>& colIdx, spMatrix_format format,
                           memory_t mem_type) {
    assert(valList.size() == colIdx.size());
    assert(dim1 + 1 == rowAccum.size());
    sparseMatrix<T>* out = nullptr;
    switch (format) {
        case SPARSEMATRIX_FORMAT_HOST_CSR:
            assert(mem_type == HOST);
            out = new hostSpMatrix_CSR<T>(dim0, dim1, valList, rowAccum, colIdx);
            break;
        case SPARSEMATRIX_FORMAT_HOST_DENSE:
            assert(mem_type == HOST);
            out = new hostSpMatrix_DENSE<T>(dim0, dim1);
            break;
#if defined(_HAS_CUDA_)
        case SPARSEMATRIX_FORMAT_CUSPARSE_DENSE:
            assert(mem_type != HOST);
            out = new cusparseSpMatrix_CSR<T>(dim0, dim1, valList, rowAccum, colIdx, mem_type);
            break;
        case SPARSEMATRIX_FORMAT_CUSPARSE_CSR:
			assert(mem_type != HOST);
            out = new cusparseSpMatrix_DENSE<T>(dim0, dim1, mem_type);
            break;
#endif
        default:
            std::fprintf(stderr, "unable to create a sparse matrix with unknown format.\n");
            break;
    }
    return out;
}

template sparseMatrix<int>* get_spMat(unsigned dim0, unsigned dim1, const std::vector<int>& valList,
                                      const std::vector<int>& rowAccum, const std::vector<int>& colIdx,
                                      spMatrix_format format, memory_t mem_type);
template sparseMatrix<float>* get_spMat(unsigned dim0, unsigned dim1, const std::vector<float>& valList,
                                        const std::vector<int>& rowAccum, const std::vector<int>& colIdx,
                                        spMatrix_format format, memory_t mem_type);
template sparseMatrix<double>* get_spMat(unsigned dim0, unsigned dim1, const std::vector<double>& valList,
                                         const std::vector<int>& rowAccum, const std::vector<int>& colIdx,
                                         spMatrix_format format, memory_t mem_type);

}  // namespace spMatrix
}  // namespace magmadnn
