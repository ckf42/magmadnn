#include "math/spgematmul.h"
namespace magmadnn {
namespace math {

template <typename T>
void spgematmul(bool trans_A, spMatrix::sparseMatrix<T>* A, bool trans_B, spMatrix::spMatrix_DENSE<T>* B,
                spMatrix::spMatrix_DENSE<T>* C) {
    assert(T_IS_SAME_MEMORY_TYPE(A, B));
    assert(T_IS_SAME_MEMORY_TYPE(A, C));
    assert(T_IS_MATRIX(B));
    assert(T_IS_MATRIX(C));

    /* op(A): MxN; op(B): NxK; C: MxK */

    unsigned C_dim_0 = C->get_shape(0), C_dim_1 = C->get_shape(1);
    assert(C_dim_0 == A->get_shape(trans_A ? 1 : 0));
    assert(C_dim_1 == B->get_shape(trans_B ? 0 : 1));
    if (A->get_data_format() == SPARSEMATRIX_FORMAT_HOST_CSR) {
        std::fprintf(stderr, "Spgemm for host_csr is not yet implemented\n");
    } else if (A->get_data_format() == SPARSEMATRIX_FORMAT_HOST_DENSE) {
        std::fprintf(stderr, "Spgemm for host_dense is not yet implemented\n");
    }
#if defined(_HAS_CUDA_)
    else {
        std::fprintf(stderr, "For spgemm on GPU please use other type-specific functions that support GPU.\n");
    }
#endif
}

#if defined(_HAS_CUDA_)

//  todo: check if cusparse can only use mat on host

//  explicit instantiation for type int, float, double
template <>
void spgematmul_cusparse<int>(bool trans_A, spMatrix::cusparseSpMatrix_CSR<int>* A, bool trans_B,
                              spMatrix::cusparseSpMatrix_DENSE<int>* B, spMatrix::cusparseSpMatrix_DENSE<int>* C,
                              spgemm_cusparse_settings settings) {
    assert(A->get_memory_type() == HOST);
    assert(B->get_memory_type() == HOST);
    assert(C->get_memory_type() == HOST);
    int alpha = 1, beta = 0;
    cusparseErrchk(cusparseSpMM(::magmadnn::internal::MAGMADNN_SETTINGS->cusparse_handle,
                                trans_A ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                                trans_B ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                *reinterpret_cast<cusparseSpMatDescr_t*>(A->get_desc()),
                                *reinterpret_cast<cusparseDnMatDescr_t*>(B->get_desc()), &beta,
                                *reinterpret_cast<cusparseDnMatDescr_t*>(C->get_desc()), CUDA_R_32I, settings.algo,
                                settings.workspace));
}

template <>
void spgematmul_cusparse<float>(bool trans_A, spMatrix::cusparseSpMatrix_CSR<float>* A, bool trans_B,
                                spMatrix::cusparseSpMatrix_DENSE<float>* B, spMatrix::cusparseSpMatrix_DENSE<float>* C,
                                spgemm_cusparse_settings settings) {
    assert(A->get_memory_type() == HOST);
    assert(B->get_memory_type() == HOST);
    assert(C->get_memory_type() == HOST);
    int alpha = 1, beta = 0;
    cusparseErrchk(cusparseSpMM(::magmadnn::internal::MAGMADNN_SETTINGS->cusparse_handle,
                                trans_A ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                                trans_B ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                *reinterpret_cast<cusparseSpMatDescr_t*>(A->get_desc()),
                                *reinterpret_cast<cusparseDnMatDescr_t*>(B->get_desc()), &beta,
                                *reinterpret_cast<cusparseDnMatDescr_t*>(C->get_desc()), CUDA_R_32F, settings.algo,
                                settings.workspace));
}

template <>
void spgematmul_cusparse<double>(bool trans_A, spMatrix::cusparseSpMatrix_CSR<double>* A, bool trans_B,
                                 spMatrix::cusparseSpMatrix_DENSE<double>* B,
                                 spMatrix::cusparseSpMatrix_DENSE<double>* C, spgemm_cusparse_settings settings) {
    assert(A->get_memory_type() == HOST);
    assert(B->get_memory_type() == HOST);
    assert(C->get_memory_type() == HOST);
    int alpha = 1, beta = 0;
    cusparseErrchk(cusparseSpMM(::magmadnn::internal::MAGMADNN_SETTINGS->cusparse_handle,
                                trans_A ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                                trans_B ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                *reinterpret_cast<cusparseSpMatDescr_t*>(A->get_desc()),
                                *reinterpret_cast<cusparseDnMatDescr_t*>(B->get_desc()), &beta,
                                *reinterpret_cast<cusparseDnMatDescr_t*>(C->get_desc()), CUDA_R_64F, settings.algo,
                                settings.workspace));
}

#endif

}  // namespace math

}  // namespace magmadnn
