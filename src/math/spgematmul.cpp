#include "math/spgematmul.h"
namespace magmadnn {
namespace math {

template <typename T>
void spgematmul(T alpha, bool trans_A, spMatrix::sparseMatrix<T>* A, bool trans_B, spMatrix::spMatrix_DENSE<T>* B,
                T beta, spMatrix::spMatrix_DENSE<T>* C, void* settings) {
    assert(T_IS_SAME_MEMORY_TYPE(A, B));
    assert(T_IS_SAME_MEMORY_TYPE(A, C));

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
    else if (A->get_data_format() == SPARSEMATRIX_FORMAT_CUSPARSE_CSR) {
        assert(B->get_data_format() == SPARSEMATRIX_FORMAT_CUSPARSE_DENSE);
        assert(C->get_data_format() == SPARSEMATRIX_FORMAT_CUSPARSE_DENSE);
        assert(settings != nullptr);
        spgematmul_cusparse<T>(alpha, trans_A, reinterpret_cast<spMatrix::cusparseSpMatrix_CSR<T>*>(A), trans_B,
                               reinterpret_cast<spMatrix::cusparseSpMatrix_DENSE<T>*>(B), trans_B, beta,
                               reinterpret_cast<spMatrix::cusparseSpMatrix_DENSE<T>*>(C),
                               reinterpret_cast<spgemm_cusparse_settings*>(settings));
    } else if (A->get_data_format() == SPARSEMATRIX_FORMAT_CUSPARSE_DENSE) {
        std::fprintf(stderr, "Spgemm for cusparse_dense is not yet implemented.\n");
    }
#endif
    else {
        std::fprintf(stderr, "No matching routine found for the input data format for spgematmul.\n");
    }
}

#if defined(_HAS_CUDA_)

template <typename T>
void spgematmul_cusparse(T alpha, bool trans_A, spMatrix::cusparseSpMatrix_CSR<T>* A, bool trans_B,
                         spMatrix::cusparseSpMatrix_DENSE<T>* B, T beta, spMatrix::cusparseSpMatrix_DENSE<T>* C,
                         spgemm_cusparse_settings settings) {
    std::fprintf(stderr, "Data type not recongnized.\n");
}
//  explicit instantiation for type int, float, double
template <>
void spgematmul_cusparse<int>(int alpha, bool trans_A, spMatrix::cusparseSpMatrix_CSR<int>* A, bool trans_B,
                              spMatrix::cusparseSpMatrix_DENSE<int>* B, int beta,
                              spMatrix::cusparseSpMatrix_DENSE<int>* C, spgemm_cusparse_settings settings) {
    cusparseErrchk(cusparseSpMM(::magmadnn::internal::MAGMADNN_SETTINGS->cusparse_handle,
                                trans_A ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                                trans_B ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                *reinterpret_cast<cusparseSpMatDescr_t*>(A->get_descriptor()),
                                *reinterpret_cast<cusparseDnMatDescr_t*>(B->get_descriptor()), &beta,
                                *reinterpret_cast<cusparseDnMatDescr_t*>(C->get_descriptor()), CUDA_R_32I,
                                settings.algo, settings.workspace));
}
template <>
void spgematmul_cusparse<float>(float alpha, bool trans_A, spMatrix::cusparseSpMatrix_CSR<float>* A, bool trans_B,
                                spMatrix::cusparseSpMatrix_DENSE<float>* B, float beta,
                                spMatrix::cusparseSpMatrix_DENSE<float>* C, spgemm_cusparse_settings settings) {
    cusparseErrchk(cusparseSpMM(::magmadnn::internal::MAGMADNN_SETTINGS->cusparse_handle,
                                trans_A ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                                trans_B ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                *reinterpret_cast<cusparseSpMatDescr_t*>(A->get_descriptor()),
                                *reinterpret_cast<cusparseDnMatDescr_t*>(B->get_descriptor()), &beta,
                                *reinterpret_cast<cusparseDnMatDescr_t*>(C->get_descriptor()), CUDA_R_32F,
                                settings.algo, settings.workspace));
}
template <>
void spgematmul_cusparse<double>(double alpha, bool trans_A, spMatrix::cusparseSpMatrix_CSR<double>* A, bool trans_B,
                                 spMatrix::cusparseSpMatrix_DENSE<double>* B, double beta,
                                 spMatrix::cusparseSpMatrix_DENSE<double>* C, spgemm_cusparse_settings settings) {
    cusparseErrchk(cusparseSpMM(::magmadnn::internal::MAGMADNN_SETTINGS->cusparse_handle,
                                trans_A ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                                trans_B ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                *reinterpret_cast<cusparseSpMatDescr_t*>(A->get_descriptor()),
                                *reinterpret_cast<cusparseDnMatDescr_t*>(B->get_descriptor()), &beta,
                                *reinterpret_cast<cusparseDnMatDescr_t*>(C->get_descriptor()), CUDA_R_64F,
                                settings.algo, settings.workspace));
}
#endif

}  // namespace math

}  // namespace magmadnn
