#include "sparseMatrix/sparseMatrix.h"

namespace magmadnn {
namespace spMatrix {

//  for abstract class sparseMatrix
#if defined(DEBUG)
template <typename T>
sparseMatrix<T>::sparseMatrix(void) : descriptor(nullptr), descriptor_is_set(false), dim0(0), dim1(0) {
    fprintf(stderr, "Constructor for sparseMatrix called without parameters.\n");
}
#endif
template <typename T>
sparseMatrix<T>::sparseMatrix(Tensor<T>* adjMatrixTensorPtr, memory_t mem_type, spMatrix_format format)
    : descriptor(nullptr), descriptor_is_set(false), mem_type(mem_type), format(format) {
    assert(T_IS_MATRIX(adjMatrixTensorPtr));
    dim0 = adjMatrixTensorPtr->get_shape(0);
    dim1 = adjMatrixTensorPtr->get_shape(1);
}
template <typename T>
sparseMatrix<T>::~sparseMatrix(void) {
    if (descriptor_is_set) {
        delete descriptor;
        descriptor_is_set = false;
    }
}

//  for abstract class spMatrix_DENSE
#if defined(DEBUG)
template <typename T>
//  todo: check if ok to define Tensor with shape {0, 0}
spMatrix_DENSE<T>::spMatrix_DENSE(void) : sparseMatrix(void), data({0, 0}) {
    fprintf(stderr, "Constructor for spMatrix_DENSE called without parameters.\n");
}
#endif
template <typename T>
spMatrix_DENSE<T>::spMatrix_DENSE(const Tensor<T>* adjMatrixTensorPtr, memory_t mem_type, spMatrix_format format)
    : sparseMatrix<T>(adjMatrixTensorPtr, mem_type, format), data(adjMatrixTensorPtr->get_shape(), mem_type) {
    data.copy_from(*adjMatrixTensorPtr);
}
template <typename T>
spMatrix_DENSE<T>::~spMatrix_DENSE(void) { /* empty */
}

//  for abstract class spMatrix_CSR
#if defined(DEBUG)
template <typename T>
spMatrix_CSR<T>::spMatrix_CSR(void) : sparseMatrix(void), nnz(0), valList({0, 0}), rowCount({0, 0}), colIdx({0, 0}) : {
    fprintf(stderr, "Constructor for spMatrix_CSR called without parameters.\n");
}
#endif
template <typename T>
spMatrix_CSR<T>::spMatrix_CSR(const Tensor<T>* adjMatrixTensorPtr, memory_t mem_type, spMatrix_format format)
    : sparseMatrix<T>(adjMatrixTensorPtr, mem_type, format) {
    nnz = 0;
    std::vector<T> V;
    std::vector<int> rowAccV, colIdxV;
    rowAccV.push_back(0);
    unsigned rowCounter = 0;
    for (unsigned j = 0; j < dim1; j++) {
        rowCounter = 0;
        for (unsigned i = 0; i < dim0; i++) {
            if (adjMatrixTensorPtr->get({i, j}) != T(0)) {
                                                nonzeroEleV.push_back(adjMatrixTensorPtr->get({ i, j });
						colIdxV.push_back(i);
						++rowCounter;
            }
        }
        rowAccV.push_back(rowCounter);
        nnz += rowCounter;
    }
    valList = Tensor<T>({unsigned(1), nnz}, {ZERO, {}}, mem_type);
    rowCount = Tensor<int>({unsigned(1), dim0 + 1}, {ZERO, {}}, mem_type);
    colIdx = Tensor<int>({unsigned(1), nnz}, {ZERO, {}}, mem_type);
    //  todo: better method?
    for (unsigned idx = 0; idx < nnz; ++idx) {
        valList.set(idx, nonzeroEleV[idx]);
        colIdx.set(idx, colIdxV[idx]);
    }
    for (unsigned idx = 0; idx < dim0 + 1; ++idx) {
        rowCount.set(idx, colIdxV[idx]);
    }
}
template <typename T>
void spMatrix_CSR<T>::get_adjMat(Tensor<T>* output) const {
    T_IS_MATRIX(output);
    assert(output->get_shape[0] == dim0);
    assert(output->get_shape[1] == dim1);
    typename std::vector<T>::const_iterator valIter = valList.begin();
    std::vector<int>::const_iterator colIter = colIdx.begin();
    for (std::vector<int>::size_type rowIdx = 0; rowIdx < dim0; ++rowIdx) {
        unsigned toFill = rowCount[rowIdx + 1] - rowCount[rowIdx];
        for (unsigned itemCount = 0; itemCount < toFill; ++toFill) {
            output->set({rowIdx, *colIter}, *valIter);
            valIter++;
            colIter++;
        }
    }
}
template <typename T>
spMatrix_CSR<T>::~spMatrix_CSR(void) {
    delete valPtr;
    delete rowPtr;
    delete colIdx;
}

//  for concrete class hostSpMatrix_DENSE
template <typename T>
hostSpMatrix_DENSE<T>::hostSpMatrix_DENSE(const Tensor<T>* adjMatrixTensorPtr)
    : spMatrix_DENSE<T>(adjMatrixTensorPtr, HOST, SPARSEMATRIX_FORMAT_HOST_DENSE) { /* empty */
}
//  explicit instantiation for type int, float, double
template class hostSpMatrix_DENSE<int>;
template class hostSpMatrix_DENSE<float>;
template class hostSpMatrix_DENSE<double>;

//  for concrete class hostSpMatrix_CSR
template <typename T>
hostSpMatrix_CSR<T>::hostSpMatrix_CSR(const Tensor<T>* adjMatrixTensorPtr)
    : spMatrix_CSR<T>(adjMatrixTensorPtr, HOST, SPARSEMATRIX_FORMAT_HOST_CSR) { /* empty */
}
//  explicit instantiation for type int, float, double
template class hostSpMatrix_CSR<int>;
template class hostSpMatrix_CSR<float>;
template class hostSpMatrix_CSR<double>;

#if defined(_HAS_CUDA_)

//  for concrete class cusparseSpMatrix_DENSE
template <typename T>
cusparseSpMatrix_DENSE<T>::cusparseSpMatrix_DENSE(const Tensor<T>* adjMatrixTensorPtr, memory_t mem_type)
    : spMatrix_DENSE<T>(adjMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE) {
    fprintf(stderr, "Requested template type for cusparseSpMatrix_DENSE is not supported\n");
}
template <typename T>
cusparseSpMatrix_DENSE<T>::~cusparseSpMatrix_DENSE(void) {
    if (descriptor_is_set) {
        cusparseDestroyDnMat(reinterpret_cast<cusparseDnMatDescr_t*>(descriptor));
        descriptor_is_set = false;
        //  todo: check if desctiptor is also freed (e.g. assigned nullptr)
    }
}
//  specialization
template <>
cusparseSpMatrix_DENSE<int>::cusparseSpMatrix_DENSE(const Tensor<int>* adjMatrixTensorPtr, memory_t mem_type)
    : spMatrix_DENSE<int>(adjMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE) {
    assert(adjMatrixTensorPtr->get_memory_type() != HOST);
    descriptor = new cusparseDnMatDescr_t;
    cusparseErrchk(cusparseCreateDnMat(reinterpret_cast<cusparseDnMatDescr_t*>(descriptor), dim0, dim1, dim0,
                                       const_cast<Tensor<int>*>(adjMatrixTensorPtr)->get_ptr(), CUDA_R_32I,
                                       CUSPARSE_ORDER_ROW));
    descriptor_is_set = true;
}
template <>
cusparseSpMatrix_DENSE<float>::cusparseSpMatrix_DENSE(const Tensor<float>* adjMatrixTensorPtr, memory_t mem_type)
    : spMatrix_DENSE<float>(adjMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE) {
    assert(adjMatrixTensorPtr->get_memory_type() != HOST);
    descriptor = new cusparseDnMatDescr_t;
    cusparseErrchk(cusparseCreateDnMat(reinterpret_cast<cusparseDnMatDescr_t*>(descriptor), dim0, dim1, dim0,
                                       const_cast<Tensor<float>*>(adjMatrixTensorPtr)->get_ptr(), CUDA_R_32F,
                                       CUSPARSE_ORDER_ROW));
    descriptor_is_set = true;
}
template <>
cusparseSpMatrix_DENSE<double>::cusparseSpMatrix_DENSE(const Tensor<double>* adjMatrixTensorPtr, memory_t mem_type)
    : spMatrix_DENSE<double>(adjMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_DENSE) {
    assert(adjMatrixTensorPtr->get_memory_type() != HOST);
    descriptor = new cusparseDnMatDescr_t;
    cusparseErrchk(cusparseCreateDnMat(reinterpret_cast<cusparseDnMatDescr_t*>(descriptor), dim0, dim1, dim0,
                                       const_cast<Tensor<double>*>(adjMatrixTensorPtr)->get_ptr(), CUDA_R_64F,
                                       CUSPARSE_ORDER_ROW));
    descriptor_is_set = true;
}
//  explicit instantiation for type int, float, double
template class cusparseSpMatrix_CSR<int>;
template class cusparseSpMatrix_CSR<float>;
template class cusparseSpMatrix_CSR<double>;

//  for concrete class cusparseSparseMat
template <typename T>
cusparseSpMatrix_CSR<T>::cusparseSpMatrix_CSR(const Tensor<T>* adjMatrixTensorPtr, memory_t mem_type)
    : spMatrix_CSR<T>(adjMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR) {
    fprintf(stderr, "Requested template type for cusparseSpMatrix_CSR is not supported\n");
}
template <typename T>
cusparseSpMatrix_CSR<T>::~cusparseSpMatrix_CSR(void) {
    if (descriptor_is_set) {
        cusparseDestroySpMat(reinterpret_cast<cusparseSpMatDescr_t*>(descriptor));
        descriptor_is_set = false;
        //  todo: check if desctiptor is also freed (e.g. assigned nullptr)
    }
}
//  specialization
template <>
cusparseSpMatrix_CSR<int>::cusparseSpMatrix_CSR(const Tensor<int>* adjMatrixTensorPtr, memory_t mem_type)
    : spMatrix_CSR<int>(adjMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR) {
    assert(adjMatrixTensorPtr->get_memory_type() != HOST);
    descriptor = new cusparseSpMatDescr_t;
    cusparseErrchk(cusparseCreateCsr(reinterpret_cast<cusparseSpMatDescr_t*>(descriptor), dim0, dim1, nnz,
                                     rowCount.get_ptr(), colIdx.get_ptr(), valList.get_ptr(), CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_16U, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32I));
    descriptor_is_set = true;
}
template <>
cusparseSpMatrix_CSR<float>::cusparseSpMatrix_CSR(const Tensor<float>* adjMatrixTensorPtr, memory_t mem_type)
    : spMatrix_CSR<float>(adjMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR) {
    assert(adjMatrixTensorPtr->get_memory_type() != HOST);
    descriptor = new cusparseSpMatDescr_t;
    cusparseErrchk(cusparseCreateCsr(reinterpret_cast<cusparseSpMatDescr_t*>(descriptor), dim0, dim1, nnz,
                                     rowCount.get_ptr(), colIdx.get_ptr(), valList.get_ptr(), CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_16U, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    descriptor_is_set = true;
}
template <>
cusparseSpMatrix_CSR<double>::cusparseSpMatrix_CSR(const Tensor<double>* adjMatrixTensorPtr, memory_t mem_type)
    : spMatrix_CSR<double>(adjMatrixTensorPtr, mem_type, SPARSEMATRIX_FORMAT_CUSPARSE_CSR) {
    assert(adjMatrixTensorPtr->get_memory_type() != HOST);
    descriptor = new cusparseSpMatDescr_t;
    cusparseErrchk(cusparseCreateCsr(reinterpret_cast<cusparseSpMatDescr_t*>(descriptor), dim0, dim1, nnz,
                                     rowCount.get_ptr(), colIdx.get_ptr(), valList.get_ptr(), CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_16U, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    descriptor_is_set = true;
}
//  explicit instantiation for type int, float, double
template class cusparseSpMatrix_CSR<int>;
template class cusparseSpMatrix_CSR<float>;
template class cusparseSpMatrix_CSR<double>;

#endif
}  // namespace spMatrix
}  // namespace magmadnn
