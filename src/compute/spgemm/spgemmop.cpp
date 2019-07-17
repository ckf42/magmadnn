#include "compute/spgemm/spgemmop.h"

namespace magmadnn {
namespace op {

#if defined(_HAS_CUDA_)
template <typename T>
void SpgemmOp<T>::init_cusparse_csr(void) {
    std::fprintf(stderr, "Requested type for SpgemmOp is not supported.\n");
}
template <>
void SpgemmOp<int>::init_cusparse_csr(void) {
    a_descriptor = a->get_descriptor();
    b_wrapper = new spMatrix::cusparseSpMatrix_DENSE<int>(b_tensor, MANAGED, false);
    b_descriptor = AS_TYPE(spMatrix::cusparseSpMatrix_DENSE<int>*, b_wrapper)->get_descriptor();
    ab_wrapper = new spMatrix::cusparseSpMatrix_DENSE<int>(this->output_tensor, this->mem_type, false);
    ab_descriptor = AS_TYPE(spMatrix::cusparseSpMatrix_DENSE<int>*, ab_wrapper)->get_descriptor();
    settings = new math::spgemm_cusparse_settings{CUSPARSE_CSRMM_ALG1, nullptr, 0};
    cusparseErrchk(cusparseSpMM_bufferSize(
        ::magmadnn::internal::MAGMADNN_SETTINGS->cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, *AS_TYPE(cusparseSpMatDescr_t*, a_descriptor),
        *AS_TYPE(cusparseDnMatDescr_t*, b_descriptor), &beta, *AS_TYPE(cusparseDnMatDescr_t*, ab_descriptor),
        CUDA_R_32I, CUSPARSE_CSRMM_ALG1, &AS_TYPE(math::spgemm_cusparse_settings*, settings)->workspace_size));
    cudaErrchk(cudaMalloc((void**) &AS_TYPE(math::spgemm_cusparse_settings*, settings)->workspace,
                          AS_TYPE(math::spgemm_cusparse_settings*, settings)->workspace_size));
}
template <>
void SpgemmOp<float>::init_cusparse_csr(void) {
    a_descriptor = a->get_descriptor();
    b_wrapper = new spMatrix::cusparseSpMatrix_DENSE<float>(b_tensor, MANAGED, false);
    b_descriptor = AS_TYPE(spMatrix::cusparseSpMatrix_DENSE<float>*, b_wrapper)->get_descriptor();
    ab_wrapper = new spMatrix::cusparseSpMatrix_DENSE<float>(this->output_tensor, this->mem_type, false);
    ab_descriptor = AS_TYPE(spMatrix::cusparseSpMatrix_DENSE<float>*, ab_wrapper)->get_descriptor();
    settings = new math::spgemm_cusparse_settings{CUSPARSE_CSRMM_ALG1, nullptr, 0};
    cusparseErrchk(cusparseSpMM_bufferSize(
        ::magmadnn::internal::MAGMADNN_SETTINGS->cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, *AS_TYPE(cusparseSpMatDescr_t*, a_descriptor),
        *AS_TYPE(cusparseDnMatDescr_t*, b_descriptor), &beta, *AS_TYPE(cusparseDnMatDescr_t*, ab_descriptor),
        CUDA_R_32I, CUSPARSE_CSRMM_ALG1, &AS_TYPE(math::spgemm_cusparse_settings*, settings)->workspace_size));
    cudaErrchk(cudaMalloc((void**) &AS_TYPE(math::spgemm_cusparse_settings*, settings)->workspace,
                          AS_TYPE(math::spgemm_cusparse_settings*, settings)->workspace_size));
}
template <>
void SpgemmOp<double>::init_cusparse_csr(void) {
    a_descriptor = a->get_descriptor();
    b_wrapper = new spMatrix::cusparseSpMatrix_DENSE<double>(b_tensor, MANAGED, false);
    b_descriptor = AS_TYPE(spMatrix::cusparseSpMatrix_DENSE<double>*, b_wrapper)->get_descriptor();
    ab_wrapper = new spMatrix::cusparseSpMatrix_DENSE<double>(this->output_tensor, this->mem_type, false);
    ab_descriptor = AS_TYPE(spMatrix::cusparseSpMatrix_DENSE<double>*, ab_wrapper)->get_descriptor();
    settings = new math::spgemm_cusparse_settings{CUSPARSE_CSRMM_ALG1, nullptr, 0};
    cusparseErrchk(cusparseSpMM_bufferSize(
        ::magmadnn::internal::MAGMADNN_SETTINGS->cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, *AS_TYPE(cusparseSpMatDescr_t*, a_descriptor),
        *AS_TYPE(cusparseDnMatDescr_t*, b_descriptor), &beta, *AS_TYPE(cusparseDnMatDescr_t*, ab_descriptor),
        CUDA_R_32I, CUSPARSE_CSRMM_ALG1, &AS_TYPE(math::spgemm_cusparse_settings*, settings)->workspace_size));
    cudaErrchk(cudaMalloc((void**) &AS_TYPE(math::spgemm_cusparse_settings*, settings)->workspace,
                          AS_TYPE(math::spgemm_cusparse_settings*, settings)->workspace_size));
}
#endif

template <typename T>
void SpgemmOp<T>::init(void) {
    switch (sp_mat_format) {
        case SPARSEMATRIX_FORMAT_HOST_CSR:
            std::fprintf(stderr, "SpgemmOp for sparse matrix format host_CSR is not implemented.\n");
            break;
#if defined(_HAS_CUDA_)
        case SPARSEMATRIX_FORMAT_CUSPARSE_CSR:
            init_cusparse_csr<T>(void);
            break;
#endif
        default:
            std::fprintf(stderr, "SpgemmOp for requested sparse matrix format is not implemented.\n") break;
    }
}

#if defined(_HAS_CUDA_)
template <typename T>
void SpgemmOp<T>::init_grad_cusparse_csr(Tensor<T>* grad, Tensor<T>* out){
    std::fprintf(stderr, "Requested type for SpgemmOp is not supported.\n");
}
template <>
void SpgemmOp<int>::init_grad_cusparse_csr(Tensor<int>* grad, Tensor<int>* out){
    grad_wrapper = new spMatrix::cusparseSpMatrix_DENSE<int>(grad, this->mem_type, false);
    grad_descriptor = AS_TYPE(spMatrix::cusparseSpMatrix_DENSE<int>*, grad_wrapper)->get_descriptor();
    out_wrapper = new spMatrix::cusparseSpMatrix_DENSE<int>(out, this->mem_type, false);
    out_desctiptor = AS_TYPE(spMatrix::cusparseSpMatrix_DENSE<int>*, out_wrapper)->get_descriptor();
    grad_settings = new math::spgemm_cusparse_settings{CUSPARSE_CSRMM_ALG1, nullptr, 0};
    cusparseErrchk(cusparseSpMM_bufferSize(::magmadnn::internal::MAGMADNN_SETTINGS->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, *AS_TYPE(cusparseSpMatDescr_t*, a_descriptor), *AS_TYPE(cusparseDnMatDescr_t*, grad_descriptor), &beta, *AS_TYPE(cusparseDnMatDescr_t*, out_desctiptor), CUDA_R_32I, CUSPARSE_CSRMM_ALG1, &AS_TYPE(math::spgemm_cusparse_settings*, grad_settings)->workspace_size));
    cudaErrchk(cudaMalloc((void**) &AS_TYPE(math::spgemm_cusparse_settings*, grad_settings)->workspace,
                          AS_TYPE(math::spgemm_cusparse_settings*, grad_settings)->workspace_size));
}
template <>
void SpgemmOp<float>::init_grad_cusparse_csr(Tensor<float>* grad, Tensor<float>* out){
    grad_wrapper = new spMatrix::cusparseSpMatrix_DENSE<float>(grad, this->mem_type, false);
    grad_descriptor = AS_TYPE(spMatrix::cusparseSpMatrix_DENSE<float>*, grad_wrapper)->get_descriptor();
    out_wrapper = new spMatrix::cusparseSpMatrix_DENSE<float>(out, this->mem_type, false);
    out_desctiptor = AS_TYPE(spMatrix::cusparseSpMatrix_DENSE<float>*, out_wrapper)->get_descriptor();
    grad_settings = new math::spgemm_cusparse_settings{CUSPARSE_CSRMM_ALG1, nullptr, 0};
    cusparseErrchk(cusparseSpMM_bufferSize(::magmadnn::internal::MAGMADNN_SETTINGS->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, *AS_TYPE(cusparseSpMatDescr_t*, a_descriptor), *AS_TYPE(cusparseDnMatDescr_t*, grad_descriptor), &beta, *AS_TYPE(cusparseDnMatDescr_t*, out_desctiptor), CUDA_R_32I, CUSPARSE_CSRMM_ALG1, &AS_TYPE(math::spgemm_cusparse_settings*, grad_settings)->workspace_size));
    cudaErrchk(cudaMalloc((void**) &AS_TYPE(math::spgemm_cusparse_settings*, grad_settings)->workspace,
                          AS_TYPE(math::spgemm_cusparse_settings*, grad_settings)->workspace_size));
}
template <>
void SpgemmOp<double>::init_grad_cusparse_csr(Tensor<double>* grad, Tensor<double>* out){
    grad_wrapper = new spMatrix::cusparseSpMatrix_DENSE<double>(grad, this->mem_type, false);
    grad_descriptor = AS_TYPE(spMatrix::cusparseSpMatrix_DENSE<double>*, grad_wrapper)->get_descriptor();
    out_wrapper = new spMatrix::cusparseSpMatrix_DENSE<double>(out, this->mem_type, false);
    out_desctiptor = AS_TYPE(spMatrix::cusparseSpMatrix_DENSE<double>*, out_wrapper)->get_descriptor();
    grad_settings = new math::spgemm_cusparse_settings{CUSPARSE_CSRMM_ALG1, nullptr, 0};
    cusparseErrchk(cusparseSpMM_bufferSize(::magmadnn::internal::MAGMADNN_SETTINGS->cusparse_handle, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, *AS_TYPE(cusparseSpMatDescr_t*, a_descriptor), *AS_TYPE(cusparseDnMatDescr_t*, grad_descriptor), &beta, *AS_TYPE(cusparseDnMatDescr_t*, out_desctiptor), CUDA_R_32I, CUSPARSE_CSRMM_ALG1, &AS_TYPE(math::spgemm_cusparse_settings*, grad_settings)->workspace_size));
    cudaErrchk(cudaMalloc((void**) &AS_TYPE(math::spgemm_cusparse_settings*, grad_settings)->workspace,
                          AS_TYPE(math::spgemm_cusparse_settings*, grad_settings)->workspace_size));
}
#endif

template <typename T>
void SpgemmOp<T>::init_grad(Tensor<T>* grad, Tensor<T>* out) {
    switch (sp_mat_format) {
        case SPARSEMATRIX_FORMAT_HOST_CSR:
            std::fprintf(stderr, "SpgemmOp for sparse matrix format host_CSR is not implemented.\n");
            break;
#if defined(_HAS_CUDA_)
        case SPARSEMATRIX_FORMAT_CUSPARSE_CSR:
            init_grad_cusparse_csr(grad, out);
            break;
#endif
        default:
            std::fprintf(stderr, "SpgemmOp for requested sparse matrix format is not implemented.\n");
            break;
    }
}

template <typename T>
SpgemmOp<T>::SpgemmOp(T alpha, spMatrix::sparseMatrix<T>* a, Operation<T>* b, bool copy = true, bool needs_grad = true)
    : copy(copy),
      alpha(alpha),
      a(a),
      b(b),
      b_tensor(b->get_output_tensor()),
      b_wrapper(nullptr),
      ab_wrapper(nullptr),
      grad_wrapper(nullptr),
      out_wrapper(nullptr),
      a_descriptor(a->get_descriptor()),
      b_descriptor(nullptr),
      ab_descriptor(nullptr),
      grad_descriptor(nullptr),
      out_desctiptor(nullptr),
      sp_mat_format(a->get_data_format()),
      settings(nullptr),
      grad_settings(nullptr) {
    assert(dynamic_cast<spMatrix::spMatrix_DENSE<T>*>(a) == nullptr &&
           "Sparse matrix \"a\" for SpgemmOp must be of sparse classes. ");
    assert(OP_IS_MATRIX(b));
    assert(a->get_shape(1) == b->get_output_shape(0));
    this->output_shape = {a->get_shape(0), b->get_output_shape(1)};
    this->mem_type = b->get_memory_type();
    this->output_tensor = new Tensor<T>(this->output_shape, this->mem_type);
    init();
    //  todo: other?
}

template <typename T>
Tensor<T>* SpgemmOp<T>::_eval(bool recompute) {
    assert(b_wrapper != nullptr && "SpgemmOp not initiated.\n");
    b_tensor = b->eval(recompute);
    b_wrapper->set_mat(b_tensor);
    math::spgematmul(alpha, false, a, false, b_wrapper, beta, ab_wrapper, settings);
    ab_wrapper->get_uncompressed_mat(this->output_tensor, (T) 1);  //  todo: do it without copying?
    return this->output_tensor;
}

template <typename T>
Tensor<T>* SpgemmOp<T>::_grad(Operation<T>* consumer, Operation<T>* var, Tensor<T>* grad) {
    assert(var == b);
    //  out = a^T * grad, shape = {a_dim_1, grad_dim_1}
    Tensor<T>* out = this->_grad_cache[(uintptr_t) b];
    if (out == nullptr) {  //  first time computing grad
        if (T_IS_MATRIX(grad)) {
            out = new Tensor<T>({a->get_shape(1), grad->get_shape(1)}, {ZERO, {}}, this->mem_type);
        } else if (T_IS_VECTOR(grad)) {
            out = new Tensor<T>({a->get_shape(1), (unsigned) 1}, {ZERO, {}, this->mem_type});
        } else {  //  grad is scalar?
            out = new Tensor<T>({a->get_shape(1), a->get_shape(b)}, {ZERO, {}}, this->mem_type);
        }
        init_grad(grad, out);
    }
    grad_wrapper->set_mat(grad);
    math::spgematmul(alpha, true, a, false, grad_wrapper, beta, out_wrapper, grad_settings);
    out_wrapper->get_uncompressed_mat(out, (T)1);
    return out;
}

template <typename T>
SpgemmOp<T>::~SpgemmOp(void) {
    if (b_wrapper != nullptr) {
        delete b_wrapper;
    }
    if (ab_wrapper != nullptr) {
        delete ab_wrapper;
    }
    if (grad_wrapper != nullptr) {
        delete grad_wrapper;
    }
    if (out_wrapper != nullptr) {
        delete out_wrapper;
    }
    
    if (settings != nullptr) {
#if defined(_HAS_CUDA_)
        if (sp_mat_format == SPARSEMATRIX_FORMAT_CUSPARSE_CSR){
            cudaErrchk(cudaFree(AS_TYPE(math::spgemm_cusparse_settings*, settings)->workspace));
        }
#endif
        delete settings;
    }
    if (grad_settings != nullptr){
#if defined(_HAS_CUDA_)
        if (sp_mat_format == SPARSEMATRIX_FORMAT_CUSPARSE_CSR){
            cudaErrchk(cudaFree(AS_TYPE(math::spgemm_cusparse_settings*, grad_settings)->workspace));
        }
#endif
    }
}

template <typename T>
SpgemmOp<T>* spgemm(spMatrix::spMatrix_DENSE<T>* a, Operation<T>* b, bool copy, bool needs_grad){
    return new SpgemmOp<T>(a, b, copy, needs_grad);
}

template <>
SpgemmOp<int>* spgemm(spMatrix::spMatrix_DENSE<int>* a, Operation<int>* b, bool copy, bool needs_grad);
template <>
SpgemmOp<float>* spgemm(spMatrix::spMatrix_DENSE<float>* a, Operation<float>* b, bool copy, bool needs_grad);
template <>
SpgemmOp<double>* spgemm(spMatrix::spMatrix_DENSE<double>* a, Operation<double>* b, bool copy, bool needs_grad);

}  // namespace op
}  // namespace magmadnn
