#include "compute/spgemm/spgemmop.h"

namespace magmadnn{
namespace op{

template <typename T>
void SpgemmOp<T>::init_host_desc(void){
    /* empty */
}
#if defined(_HAS_CUDA_)
template <typename T>
void SpgemmOp<T>::init_cusparse_desc(void){
    std::fprintf(stderr, "Unknown data type for SpgemmOp.\n");
}
template <>
void SpgemmOp<int>::init_cusparse_desc(void){
    // assert(OP_IS)
    a_descriptor = AS_TYPE(spMatrix::cusparseSpMatrix_CSR<int>*, a)->get_descriptor();
    b_wrapper = new spMatrix::cusparseSpMatrix_DENSE<int>(b->get_output_tensor(), MANAGED, false);
    b_descriptor = AS_TYPE(spMatrix::cusparseSpMatrix_DENSE<int>*, b)->get_descriptor();
    c_wrapper = new spMatrix::cusparseSpMatrix_DENSE<int>(c->get_output_tensor(), MANAGED, false);
    c_descriptor = AS_TYPE(spMatrix::cusparseSpMatrix_DENSE<int>*, c)->get_descriptor();
    settings = new math::spgemm_cusparse_settings;
    AS_TYPE(math::spgemm_cusparse_settings*, settings)->algo = CUSPARSE_CSRMM_ALG1;
    cusparseErrchk(cusparseSpMM_bufferSize(::magmadnn::internal::MAGMADNN_SETTINGS->cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, *AS_TYPE(cusparseSpMatDescr_t*, a_descriptor), *AS_TYPE(cusparseDnMatDescr_t*, b_descriptor), &beta, *AS_TYPE(cusparseDnMatDescr_t*, c_descriptor), CUDA_R_32I, AS_TYPE(math::spgemm_cusparse_settings*, settings)->algo, &(AS_TYPE(math::spgemm_cusparse_settings*, settings)->workspace_size)));
}
#endif
template <typename T>
void SpgemmOp<T>::init_desc(void){
    if (a->get_data_format == SPARSEMATRIX_FORMAT_HOST_CSR){

    }
    #if defined(_HAS_CUDA_)
    else if (a->get_data_format == SPARSEMATRIX_FORMAT_CUSPARSE_CSR){

    }
    #endif
    else {
        std::fprintf(stderr, "Input sparse matrix a in spgemmOp is not of sparse format\n");
    }
}

template <typename T>
SpgemmOp<T>::SpgemmOp(T alpha, spMatrix::sparseMatrix<T>* a, Operation<T>* b, T beta, Operation<T>* c, bool copy = true,bool needs_grad = true): copy(copy), alpha(alpha), beta(beta), a(a), b(b), c(c), b_wrapper(nullptr), c_wrapper(nullptr), sp_mat_format(a->get_data_format()), settings(nullptr) {
    assert()
    init_desc();

}

template <typename T>
SpgemmOp<T>::~SpgemmOp(void){
    if (b_wrapper != nullptr){delete b_wrapper;}
    if (c_wrapper != nullptr){delete c_wrapper;}
    if (settings != nullptr){delete settings;}
}


}  //  namespace magmadnn
}  //  namespace op
