#include "math/spgematmul.h"
namespace magmadnn {
namespace math {

//  todo: clean up

template <typename T>
void spgematmul(T alpha, bool trans_A, spMatrix::sparseMatrix<T>* A, bool trans_B, spMatrix::spMatrix_DENSE<T>* B,
                T beta, spMatrix::spMatrix_DENSE<T>* C, void* settings, bool col_major_output) {
    assert(T_IS_SAME_MEMORY_TYPE(A, B));
    assert(T_IS_SAME_MEMORY_TYPE(A, C));

    /* op(A): MxN; op(B): NxK; C: MxK */

    unsigned C_dim_0 = C->get_shape(0), C_dim_1 = C->get_shape(1);
    assert(C_dim_0 == A->get_shape(trans_A ? 1 : 0));
    assert(C_dim_1 == B->get_shape(trans_B ? 0 : 1));
    if (A->get_data_format() == SPARSEMATRIX_FORMAT_HOST_CSR) {
        std::fprintf(stderr, "Spgemm for host_csr is not yet implemented\n");
    } else if (A->get_data_format() == SPARSEMATRIX_FORMAT_HOST_DENSE) {
        std::fprintf(stderr, "For matrix of dense format please use matmul\n");
    }
#if defined(_HAS_CUDA_)
    else if (A->get_data_format() == SPARSEMATRIX_FORMAT_CUSPARSE_CSR) {
        assert(B->get_data_format() == SPARSEMATRIX_FORMAT_CUSPARSE_DENSE);
        assert(C->get_data_format() == SPARSEMATRIX_FORMAT_CUSPARSE_DENSE);
#if (CUDART_VERSION >= 100100)
        assert(settings != nullptr);
        spgematmul_cusparse<T>(alpha, trans_A, A, trans_B, AS_TYPE(spMatrix::cusparseSpMatrix_DENSE<T>*, B), beta,
                               AS_TYPE(spMatrix::cusparseSpMatrix_DENSE<T>*, C),
                               *AS_TYPE(spgemm_cusparse_settings*, settings), col_major_output);
#elif (CUDART_VERSION < 100100)
        spgematmul_cusparse_csr<T>(alpha, trans_A, AS_TYPE(spMatrix::cusparseSpMatrix_CSR<T>*, A), trans_B,
                                   AS_TYPE(spMatrix::cusparseSpMatrix_DENSE<T>*, B), beta,
                                   AS_TYPE(spMatrix::cusparseSpMatrix_DENSE<T>*, C), col_major_output);
#endif
    } else if (A->get_data_format() == SPARSEMATRIX_FORMAT_CUSPARSE_DENSE) {
        std::fprintf(stderr, "For matrix of dense format please use matmul\n");
    }
#endif
    else {
        std::fprintf(stderr, "No matching routine found for the input data format for spgematmul.\n");
    }
}

template void spgematmul<int>(int alpha, bool trans_A, spMatrix::sparseMatrix<int>* A, bool trans_B,
                              spMatrix::spMatrix_DENSE<int>* B, int beta, spMatrix::spMatrix_DENSE<int>* C,
                              void* settings, bool col_major_output);
template void spgematmul<float>(float alpha, bool trans_A, spMatrix::sparseMatrix<float>* A, bool trans_B,
                                spMatrix::spMatrix_DENSE<float>* B, float beta, spMatrix::spMatrix_DENSE<float>* C,
                                void* settings, bool col_major_output);
template void spgematmul<double>(double alpha, bool trans_A, spMatrix::sparseMatrix<double>* A, bool trans_B,
                                 spMatrix::spMatrix_DENSE<double>* B, double beta, spMatrix::spMatrix_DENSE<double>* C,
                                 void* settings, bool col_major_output);

#if defined(_HAS_CUDA_)
#if (CUDART_VERSION >= 100100)
template <typename T>
void spgematmul_cusparse(T alpha, bool trans_A, spMatrix::sparseMatrix<T>* A, bool trans_B,
                         spMatrix::cusparseSpMatrix_DENSE<T>* B, T beta, spMatrix::cusparseSpMatrix_DENSE<T>* C,
                         spgemm_cusparse_settings settings, bool col_major_output) {
    cusparseErrchk(cusparseSpMM(::magmadnn::internal::MAGMADNN_SETTINGS->cusparse_handle,
                                trans_A ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE,
                                trans_B ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE, &alpha,
                                *AS_TYPE(cusparseSpMatDescr_t*, A->get_descriptor()),
                                *AS_TYPE(cusparseDnMatDescr_t*, B->get_descriptor()), &beta,
                                *AS_TYPE(cusparseDnMatDescr_t*, C->get_descriptor()), B->get_data_type(), settings.algo,
                                settings.workspace));
    //  now C->_data stores ass column-major
    if (!col_major_output) {
        internal::transpose_full_device<T>(C->get_data_ptr(), C->get_data_ptr());
        //  now C->_data stores as a row-major
    }
}
template void spgematmul_cusparse<int>(int alpha, bool trans_A, spMatrix::sparseMatrix<int>* A, bool trans_B,
                                       spMatrix::cusparseSpMatrix_DENSE<int>* B, int beta,
                                       spMatrix::cusparseSpMatrix_DENSE<int>* C, spgemm_cusparse_settings settings,
                                       bool col_major_output);
template void spgematmul_cusparse<float>(float alpha, bool trans_A, spMatrix::sparseMatrix<float>* A, bool trans_B,
                                         spMatrix::cusparseSpMatrix_DENSE<float>* B, float beta,
                                         spMatrix::cusparseSpMatrix_DENSE<float>* C, spgemm_cusparse_settings settings,
                                         bool col_major_output);
template void spgematmul_cusparse<double>(double alpha, bool trans_A, spMatrix::sparseMatrix<double>* A, bool trans_B,
                                          spMatrix::cusparseSpMatrix_DENSE<double>* B, double beta,
                                          spMatrix::cusparseSpMatrix_DENSE<double>* C,
                                          spgemm_cusparse_settings settings, bool col_major_output);
#elif (CUDART_VERSION < 100100)
template <typename T>
void spgematmul_cusparse_csr(T alpha, bool trans_A, spMatrix::cusparseSpMatrix_CSR<T>* A, bool trans_B,
                             spMatrix::cusparseSpMatrix_DENSE<T>* B, T beta, spMatrix::cusparseSpMatrix_DENSE<T>* C,
                             bool col_major_output) {
    std::fprintf(stderr, "Data type not recongnized.\n");
}
template <>
void spgematmul_cusparse_csr(float alpha, bool trans_A, spMatrix::cusparseSpMatrix_CSR<float>* A, bool trans_B,
                             spMatrix::cusparseSpMatrix_DENSE<float>* B, float beta,
                             spMatrix::cusparseSpMatrix_DENSE<float>* C, bool col_major_output) {
    if (!trans_B){
        assert(trans_A != true && "cusparse csrmm2 does not support such transpose combination.");
    }
    cusparseErrchk(cusparseScsrmm2(
        ::magmadnn::internal::MAGMADNN_SETTINGS->cusparse_handle, 
        trans_A?CUSPARSE_OPERATION_TRANSPOSE:CUSPARSE_OPERATION_NON_TRANSPOSE, 
        trans_B?CUSPARSE_OPERATION_NON_TRANSPOSE:CUSPARSE_OPERATION_TRANSPOSE, 
        A->get_shape(0), B->get_shape(trans_B?0:1), A->get_shape(1), A->get_nnz(), 
        &alpha, *AS_TYPE(cusparseMatDescr_t*, A->get_descriptor()), 
        A->get_val_ptr()->get_ptr(), A->get_row_ptr()->get_ptr(), A->get_col_ptr()->get_ptr(), 
        B->get_data_ptr()->get_ptr(), B->get_shape(1), 
        &beta, C->get_data_ptr()->get_ptr(), C->get_shape(0)
        ));
    if (!col_major_output) {
        C->get_data_ptr()->reshape({C->get_shape(1), C->get_shape(0)});
        internal::transpose_full_device(C->get_data_ptr(), C->get_data_ptr());
        C->get_data_ptr()->reshape({C->get_shape(0), C->get_shape(1)});
    }
}
template <>
void spgematmul_cusparse_csr(double alpha, bool trans_A, spMatrix::cusparseSpMatrix_CSR<double>* A, bool trans_B,
                             spMatrix::cusparseSpMatrix_DENSE<double>* B, double beta,
                             spMatrix::cusparseSpMatrix_DENSE<double>* C, bool col_major_output) {
    if (!trans_B){
        assert(trans_A != true && "cusparse csrmm2 does not support such transpose combination.");
    }
    cusparseErrchk(cusparseDcsrmm2(
        ::magmadnn::internal::MAGMADNN_SETTINGS->cusparse_handle, 
        trans_A?CUSPARSE_OPERATION_TRANSPOSE:CUSPARSE_OPERATION_NON_TRANSPOSE, 
        trans_B?CUSPARSE_OPERATION_NON_TRANSPOSE:CUSPARSE_OPERATION_TRANSPOSE, 
        A->get_shape(0), B->get_shape(trans_B?0:1), A->get_shape(1), A->get_nnz(), 
        &alpha, *AS_TYPE(cusparseMatDescr_t*, A->get_descriptor()), 
        A->get_val_ptr()->get_ptr(), A->get_row_ptr()->get_ptr(), A->get_col_ptr()->get_ptr(), 
        B->get_data_ptr()->get_ptr(), B->get_shape(1), 
        &beta, C->get_data_ptr()->get_ptr(), C->get_shape(0)
        ));
    if (!col_major_output) {
        C->get_data_ptr()->reshape({C->get_shape(1), C->get_shape(0)});
        internal::transpose_full_device(C->get_data_ptr(), C->get_data_ptr());
        C->get_data_ptr()->reshape({C->get_shape(0), C->get_shape(1)});
    }
}
#endif
#endif

}  // namespace math

}  // namespace magmadnn
