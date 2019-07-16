#include "compute/spgemm/spgemmop.h"

namespace magmadnn{
namespace op{


template <typename T>
void spgemmOp<T>::init_desc(void){
    std::fprintf(stderr, "Unknown data type passed");
}

#if defined(_HAS_CUDA_)
template <typename T>
void spgemmOp<T>::init_cusparse_desc(void){
    std::fprintf(stderr, "Unknown data type passed");
}

template <>
void spgemmOp<int>::init_cusparse_desc(void){
    if (mat_format == SPARSEMATRIX_FORMAT_CUSPARSE_CSR || mat_format == SPARSEMATRIX_FORMAT_CUSPARSE_DENSE){
        //  use cusparse descriptors
        b_wrapper = new cusparseDnMatDescr_t;
        c_wrapper = new cusparseDnMatDescr_t;
        cusparseErrchk(cusparseCreateDnMat(reinterpret_cast<cusparseDnMatDescr_t*>(b_wrapper), b->get_output_shape(0), b->get_output_shape(1), b->get_output_shape(0), b->get_output_tensor()->get_ptr(), CUDA_R_32I, CUSPARSE_ORDER_ROW));
        cusparseErrchk(cusparseCreateDnMat(reinterpret_cast<cusparseDnMatDescr_t*>(c_wrapper), c->get_output_shape(0), c->get_output_shape(1), c->get_output_shape(0), c->get_output_tensor()->get_ptr(), CUDA_R_32I, CUSPARSE_ORDER_ROW));
        cusparse_settings.algo = CUSPARSE_CSRMM_ALG1;
        cusparseErrchk(cusparseSpMM_bufferSize(::magmadnn::internal::MAGMADNN_SETTINGS->cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, *reinterpret_cast<cusparseSpMatDescr_t*>(a->get_desc()), *reinterpret_cast<cusparseDnMatDescr_t*>(b_wrapper), &beta, *reinterpret_cast<cusparseDnMatDescr_t*>(c_wrapper), CUDA_R_32I, cusparse_settings.algo, &cusparse_settings.workspace_size));
    }
}
template <>
void spgemmOp<float>::init_cusparse_desc(void){
    if (mat_format == SPARSEMATRIX_FORMAT_CUSPARSE_CSR || mat_format == SPARSEMATRIX_FORMAT_CUSPARSE_DENSE){
        //  use cusparse descriptors
        b_wrapper = new cusparseDnMatDescr_t;
        c_wrapper = new cusparseDnMatDescr_t;
        cusparseErrchk(cusparseCreateDnMat(reinterpret_cast<cusparseDnMatDescr_t*>(b_wrapper), b->get_output_shape(0), b->get_output_shape(1), b->get_output_shape(0), b->get_output_tensor()->get_ptr(), CUDA_R_32F, CUSPARSE_ORDER_ROW));
        cusparseErrchk(cusparseCreateDnMat(reinterpret_cast<cusparseDnMatDescr_t*>(c_wrapper), c->get_output_shape(0), c->get_output_shape(1), c->get_output_shape(0), c->get_output_tensor()->get_ptr(), CUDA_R_32F, CUSPARSE_ORDER_ROW));
        cusparse_settings.algo = CUSPARSE_CSRMM_ALG1;
        cusparseErrchk(cusparseSpMM_bufferSize(::magmadnn::internal::MAGMADNN_SETTINGS->cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, *reinterpret_cast<cusparseSpMatDescr_t*>(a->get_desc()), *reinterpret_cast<cusparseDnMatDescr_t*>(b_wrapper), &beta, *reinterpret_cast<cusparseDnMatDescr_t*>(c_wrapper), CUDA_R_32F, cusparse_settings.algo, &cusparse_settings.workspace_size));
    }
}
template <>
void spgemmOp<double>::init_cusparse_desc(void){
    if (mat_format == SPARSEMATRIX_FORMAT_CUSPARSE_CSR || mat_format == SPARSEMATRIX_FORMAT_CUSPARSE_DENSE){
        //  use cusparse descriptors
        b_wrapper = new cusparseDnMatDescr_t;
        c_wrapper = new cusparseDnMatDescr_t;
        cusparseErrchk(cusparseCreateDnMat(reinterpret_cast<cusparseDnMatDescr_t*>(b_wrapper), b->get_output_shape(0), b->get_output_shape(1), b->get_output_shape(0), b->get_output_tensor()->get_ptr(), CUDA_R_64F, CUSPARSE_ORDER_ROW));
        cusparseErrchk(cusparseCreateDnMat(reinterpret_cast<cusparseDnMatDescr_t*>(c_wrapper), c->get_output_shape(0), c->get_output_shape(1), c->get_output_shape(0), c->get_output_tensor()->get_ptr(), CUDA_R_64F, CUSPARSE_ORDER_ROW));
        cusparse_settings.algo = CUSPARSE_CSRMM_ALG1;
        cusparseErrchk(cusparseSpMM_bufferSize(::magmadnn::internal::MAGMADNN_SETTINGS->cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, *reinterpret_cast<cusparseSpMatDescr_t*>(a->get_desc()), *reinterpret_cast<cusparseDnMatDescr_t*>(b_wrapper), &beta, *reinterpret_cast<cusparseDnMatDescr_t*>(c_wrapper), CUDA_R_64F, cusparse_settings.algo, &cusparse_settings.workspace_size));
    }
}
#endif

template <typename T>
void spgemmOp<T>::init_desc(void){
    switch (){
#if defined(_HAS_CUDA_)
    case SPARSEMATRIX_FORMAT_CUSPARSE_CSR:
    init_cusparse_desc();
    break;
#endif
    default:
    fprintf(stderr, "Unknown sparse matrix data format.\n");
    break;
    }
}

template <typename T>
spgemmOp<T>::spgemmOp(T alpha, spMatrix::sparseMatrix<T>* a, Operation<T>* b, T beta, Operation<T> *c, bool copy, bool needs_grad): Operation<T>::Operation({b, c}, needs_grad), alpha(alpha), a(a), b(b), beta(beta), c(c), copy(copy), b_wrapper(nullptr), c_wrapper(nullptr), mat_format(a->get_data_format()){
    //  same memory type
    assert(a->get_memory_type() == b->get_memory_type());
    assert(c->get_memory_type() == b->get_memory_type());
    //  must be matrices
    OP_IS_MATRIX(b);
    OP_IS_MATRIX(c);
    //  a: MxK, b: KxN, c:MxN
    unsigned M = a->get_shape(0), K = b->get_output_shape(0), N = c->get_output_shape(1);
    //  valid shape
    assert(M == c->get_output_shape(0));
    assert(K == a->get_shape(1));
    assert(N == b->get_output_shape(1));

    this->output_shape = {M, N};
    this->mem_type = a->get_memory_type();

    if (copy){
        this->output_tensor = new Tensor<T>(this->output_shape, this->mem_type);
    }
    this->_grad_cache[(uintptr_t) b] = NULL;
    this->_grad_cache[(uintptr_t) c] = NULL;

    init_desc();

}

template <typename T>
spgemmOp<T>::~spgemmOp(void){
    #if defined(_HAS_CUDA_)
    if (mat_format == SPARSEMATRIX_FORMAT_CUSPARSE_CSR || mat_format == SPARSEMATRIX_FORMAT_CUSPARSE_DENSE){
        cusparseDestroyDnMat(b_wrapper);
        cusparseDestroyDnMat(c_wrapper);
    }
    #endif
}

template <typename T>
Tensor<T>* spgemmOp<T>::_eval(bool recompute){
    b_tensor = b->eval(recompute);
    c_tensor = c->eval(recompute);
    if (copy){
        this->output_tensor->copy_form(*c_tensor);
    } else {
        this->output_tensor = c_tensor;
    }

    switch (mat_format){
        #if defined(_HAS_CUDA_)
        case SPARSEMATRIX_FORMAT_CUSPARSE_CSR:
        math::spgematmul_cusparse(false, reinterpret_cast<spMatrix::cusparseSpMatrix_CSR<T>*>(a), false, reinterpret_cast<spMatrix::cusparseSpMatrix_DENSE*>(b_wrapper), reinterpret_cast<spMatrix::cusparseSpMatrix_DENSE*>(c_wrapper), cusparse_settings);
        break;
        #endif
        default:
        math::spgematmul(false, a, false, )
        break;
    }
    return this->output_tensor;
}

template <typename T>
Tensor<T> *spgemmOp<T>::_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad){
    Tensor<T>* out = this->_grad_cache[(uintptr_t) var];
    b_tensor = b->eval(false);
    if (out == NULL){
        if (T_IS_MATRIX(grad)){
            // grad = a->get_uncompressed_mat()^T * grad, grad is matrix
            out = new Tensor<T>({a->get_shape(1), grad->get_shape(0)}, {NONE, {}}, this->mem_type);
            math::spgematmul(true, a, false, ?, ?);
        } else if (T_IS_VECTOR(grad)){
            // grad = a->get_uncompressed_mat()^T * grad, grad is vector
            out = new Tensor<T>({a->get_shape(1)}, {NONE, {}}, this->mem_type);
            
        } else {
            // grad = a->get_uncompressed_mat()^T * grad, grad is scalar?
            out = new Tensor<T>({a->get_shape(1), a->get_shape(0)}, {NONE, {}}, this->mem_type);
        }
    }
    return out;
}


}  //  namespace magmadnn
}  //  namespace op