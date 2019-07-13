#include "sparseMatrix/sparseMatrix.h"

namespace magmadnn {

template <typename T>
baseSparseMat_CSR<T>::baseSparseMat_CSR(const Tensor<T>* adjMatrixTensorPtr, memory_t mem_type) {
    nnz = 0;
    std::vector<T> nonzeroEle;
    std::vector<int> rowAccV, colIdxV;
    rowAccV.push_back(0);
    unsigned rowCounter = 0;
    for (unsigned j = 0; j < dim2; j++) {
        rowCounter = 0;
        for (unsigned i = 0; i < dim1; i++) {
            if (adjMatrixTensorPtr->get({i, j}) != T(0)) {
                    nonzeroEle.push_back(adjMatrixTensorPtr->get({ i, j });
					colIdxV.push_back(i);
					++rowCounter;
            }
        }
        rowAccV.push_back(rowCounter);
        nnz += rowCounter;
    }
    valPtr = new Tensor<T>({unsigned(1), nnz}, {ZERO, {}}, mem_type);
    rowPtr = new Tensor<int>({unsigned(1), dim1 + 1}, {ZERO, {}}, mem_type);
    colIdx = new Tensor<int>({unsigned(1), nnz}, {ZERO, {}}, mem_type);
    for (unsigned idx = 0; idx < nnz; ++idx) {
        valPtr->set(idx, nonzeroEle[idx]);
        colIdx->set(idx, colIdxV[idx]);
    }
    for (unsigned idx = 0; idx < dim1 + 1; ++idx) {
        rowPtr->set(idx, colIdxV[idx]);
    }
}

/* template <typename T>
Tensor<T> baseSparseMat_CSR<T>::get_uncompressed_adjMat(void) const {

} */

template <typename T>
baseSparseMat_CSR<T>::~baseSparseMat_CSR(void) {
    delete valPtr;
    delete rowPtr;
    delete colIdx;
}

template <typename T>
hostSparseMat_CSR<T>::hostSparseMat_CSR(const Tensor<T>* adjMatrixTensorPtr)
    : baseSparseMat_CSR<T>(adjMatrixTensorPtr, HOST) {
    format = HOST_CSR;
}

template class hostSparseMat_CSR<int>;
template class hostSparseMat_CSR<float>;
template class hostSparseMat_CSR<double>;

#if defined(_HAS_CUDA_)
template <typename T>
cusparseMat_CSR<T>::cusparseMat_CSR(const Tensor<T>* adjMatrixTensorPtr, memory_t mem_type) {
    printf("Requested template type is not supported\n");
}

template <typename T>
cusparseMat_CSR<T>::~cusparseMat_CSR(void) {
    if (descripter_is_set) {
        cusparseDestroySpMat((cusparseSpMatDescr_t*) descripter);
        descripter_is_set = false;
    }
}

template <>
cusparseMat_CSR<int>::cusparseMat_CSR(const Tensor<int>* adjMatrixTensorPtr, memory_t mem_type)
    : baseSparseMat_CSR<int>(adjMatrixTensorPtr, mem_type) {
    format = CUSPARSE_CSR;
    descripter = new cusparseSpMatDescr_t;
    cusparseErrchk(cusparseCreateCsr((cusparseSpMatDescr_t*) descripter, dim1, dim2, nnz, rowPtr->get_ptr(),
                                     colIdx->get_ptr(), valPtr->get_ptr(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32I));
    descripter_is_set = true;
}

template <>
cusparseMat_CSR<float>::cusparseMat_CSR(const Tensor<float>* adjMatrixTensorPtr, memory_t mem_type)
    : baseSparseMat_CSR<float>(adjMatrixTensorPtr, mem_type) {
    format = CUSPARSE_CSR;
    cusparseErrchk(cusparseCreateCsr((cusparseSpMatDescr_t*) descripter, dim1, dim2, nnz, rowPtr->get_ptr(),
                                     colIdx->get_ptr(), valPtr->get_ptr(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    descripter_is_set = true;
}

template <>
cusparseMat_CSR<double>::cusparseMat_CSR(const Tensor<double>* adjMatrixTensorPtr, memory_t mem_type)
    : baseSparseMat_CSR<double>(adjMatrixTensorPtr, mem_type) {
    format = CUSPARSE_CSR;
    cusparseErrchk(cusparseCreateCsr((cusparseSpMatDescr_t*) descripter, dim1, dim2, nnz, rowPtr->get_ptr(),
                                     colIdx->get_ptr(), valPtr->get_ptr(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    descripter_is_set = true;
}

template class cusparseMat_CSR<int>;
template class cusparseMat_CSR<float>;
template class cusparseMat_CSR<double>;

#endif

}  // namespace magmadnn