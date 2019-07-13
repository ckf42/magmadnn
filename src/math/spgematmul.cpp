#include "math/spgematmul.h"
namespace magmadnn {
namespace math {

template <>
void spgematmul(int alpha, bool trans_A, sparseMatrix<int> *A, bool trans_B, Tensor<int> *B, int beta, Tensor<int> *C);

template <>
void spgematmul(float alpha, bool trans_A, sparseMatrix<float> *A, bool trans_B, Tensor<float> *B, float beta,
                Tensor<float> *C) {
    assert(T_IS_SAME_MEMORY_TYPE(A, B));
    assert(T_IS_SAME_MEMORY_TYPE(A, C));
    assert(T_IS_MATRIX(B));
    assert(T_IS_MATRIX(C));

    /* op(A): MxN; op(B): NxK; C: MxK */

    unsigned C_dim_1 = C->get_shape(0), C_dim_2 = C->get_shape(1);
    assert(C_dim_1 == A->get_shape(trans_A ? 1 : 0));
    assert(C_dim_2 == B->get_shape(trans_B ? 0 : 1));
    if (A->get_format()) {
        std::fprintf(stderr, "Spgemm for host_csr is not yet implemented\n");
    }
#if defined(_HAS_CUDA_)
    else if (A->get_format() == CUSPARSE_CSR) {
        size_t bufferSize;
        cusparseHandle_t handle;
        // cusparseErrchk(cusparseCreate(&handle));
        cusparseErrchk(cusparseSpMM_bufferSize(?, trans_A?CUSPARSE_OPERATION_TRANSPOSE:CUSPARSE_OPERATION_NON_TRANSPOSE, trans_B?CUSPARSE_OPERATION_TRANSPOSE:CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, *(cusparseSpMatDescr_t*)A->get_desc(), ?, &beta, ?, CUDA_R_32F, CUSPARSE_CSRMM_ALG1, &buffersize);
        // cusperrchk(cusparseDestroy(handle));
    }
#endif
    else {
        std::fprintf(stderr, "Spgemm for data format of A is not implemented.\n");
    }
}

}  // namespace math

}  // namespace magmadnn
}  // namespace magmadnn