#include "compute/gcnconv/gcnconvop_internal.h"

namespace magmadnn {
namespace internal {

#if defined(_HAS_CUDA_)
template <typename T>
void set_cusparse_spmm_settings(void*& settings, cudaDataType data_type, const T* alpha, bool spMatDoTrans,
                                spMatrix::sparseMatrix<T>* spMat, bool dnMatDoTrans, spMatrix::spMatrix_DENSE<T>* dnMat,
                                const T* beta, spMatrix::spMatrix_DENSE<T>* dnOut, cusparseSpMMAlg_t alg) {
    settings = new math::spgemm_cusparse_settings{alg, nullptr, 0};
    cusparseErrchk(
        cusparseSpMM_bufferSize(::magmadnn::internal::MAGMADNN_SETTINGS->cusparse_handle,
                                spMatDoTrans ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                                dnMatDoTrans ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE, alpha,
                                *AS_TYPE(cusparseSpMatDescr_t*, spMat->get_descriptor()),
                                *AS_TYPE(cusparseDnMatDescr_t*, dnMat->get_descriptor()), beta,
                                *AS_TYPE(cusparseDnMatDescr_t*, dnOut->get_descriptor()), data_type, alg,
                                &AS_TYPE(math::spgemm_cusparse_settings*, settings)->workspace_size));
    cudaErrchk(cudaMalloc((void**) &AS_TYPE(math::spgemm_cusparse_settings*, settings)->workspace,
                          AS_TYPE(math::spgemm_cusparse_settings*, settings)->workspace_size));
}
template void set_cusparse_spmm_settings<int>(void*& settings, cudaDataType data_type, const int* alpha,
                                              bool spMatDoTrans, spMatrix::sparseMatrix<int>* spMat, bool dnMatDoTrans,
                                              spMatrix::spMatrix_DENSE<int>* dnMat, const int* beta,
                                              spMatrix::spMatrix_DENSE<int>* dnOut, cusparseSpMMAlg_t alg);
template void set_cusparse_spmm_settings<float>(void*& settings, cudaDataType data_type, const float* alpha,
                                                bool spMatDoTrans, spMatrix::sparseMatrix<float>* spMat,
                                                bool dnMatDoTrans, spMatrix::spMatrix_DENSE<float>* dnMat,
                                                const float* beta, spMatrix::spMatrix_DENSE<float>* dnOut,
                                                cusparseSpMMAlg_t alg);
template void set_cusparse_spmm_settings<double>(void*& settings, cudaDataType data_type, const double* alpha,
                                                 bool spMatDoTrans, spMatrix::sparseMatrix<double>* spMat,
                                                 bool dnMatDoTrans, spMatrix::spMatrix_DENSE<double>* dnMat,
                                                 const double* beta, spMatrix::spMatrix_DENSE<double>* dnOut,
                                                 cusparseSpMMAlg_t alg);

#endif

}  //  namespace internal
}  //  namespace magmadnn
