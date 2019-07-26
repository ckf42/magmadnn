#include "compute/gcnconv/gcnconvop.h"
//  todo: clean up (may need total rewrite)
namespace magmadnn {
namespace op {

#if defined(_HAS_CUDA_)

template <typename T>
void GCNConvOp<T>::cublasGemmStridedBatchedWrap(bool trans_A, bool trans_B, int m, int n, int k, T alpha, T* A, int lda,
                                                long long strideA, T* B, int ldb, long long strideB, T beta, T* C,
                                                int ldc, long long strideC, int batch) {
    std::fprintf(stderr, "Data type not supported by cublas.\n");
}
template <>
void GCNConvOp<float>::cublasGemmStridedBatchedWrap(bool trans_A, bool trans_B, int m, int n, int k, float alpha,
                                                    float* A, int lda, long long strideA, float* B, int ldb,
                                                    long long strideB, float beta, float* C, int ldc, long long strideC,
                                                    int batch) {
    cublasErrchk(cublasSgemmStridedBatched(::magmadnn::internal::MAGMADNN_SETTINGS->cublas_handle,
                                           trans_A ? CUBLAS_OP_T : CUBLAS_OP_N, trans_B ? CUBLAS_OP_T : CUBLAS_OP_N, m,
                                           n, k, &alpha, A, lda, strideA, B, ldb, strideB, &beta, C, ldc, strideC,
                                           batch));
}
template <>
void GCNConvOp<double>::cublasGemmStridedBatchedWrap(bool trans_A, bool trans_B, int m, int n, int k, double alpha,
                                                     double* A, int lda, long long strideA, double* B, int ldb,
                                                     long long strideB, double beta, double* C, int ldc,
                                                     long long strideC, int batch) {
    cublasErrchk(cublasDgemmStridedBatched(::magmadnn::internal::MAGMADNN_SETTINGS->cublas_handle,
                                           trans_A ? CUBLAS_OP_T : CUBLAS_OP_N, trans_B ? CUBLAS_OP_T : CUBLAS_OP_N, m,
                                           n, k, &alpha, A, lda, strideA, B, ldb, strideB, &beta, C, ldc, strideC,
                                           batch));
}

#if (CUDART_VERSION >= 100100)
template <typename T>
void GCNConvOp<T>::init_cusparse_settings(cusparseSpMMAlg_t alg) {
    std::fprintf(stderr, "Requested data type for GCNConvOp is not supported.\n");
}
template <>
void GCNConvOp<int>::init_cusparse_settings(cusparseSpMMAlg_t alg) {
    internal::set_cusparse_spmm_settings(ab_settings, CUDA_R_32I, &const_one, false, a, false, b_wrapper, &const_zero,
                                         ab_wrapper, alg);
}
template <>
void GCNConvOp<float>::init_cusparse_settings(cusparseSpMMAlg_t alg) {
    internal::set_cusparse_spmm_settings(ab_settings, CUDA_R_32F, &const_one, false, a, false, b_wrapper, &const_zero,
                                         ab_wrapper, alg);
}
template <>
void GCNConvOp<double>::init_cusparse_settings(cusparseSpMMAlg_t alg) {
    internal::set_cusparse_spmm_settings(ab_settings, CUDA_R_64F, &const_one, false, a, false, b_wrapper, &const_zero,
                                         ab_wrapper, alg);
}

template <typename T>
void GCNConvOp<T>::init_aTgrad_cusparse_settings(cusparseSpMMAlg_t alg) {
    std::fprintf(stderr, "Requested data type for GCNConvOp is not supported.\n");
}
template <>
void GCNConvOp<int>::init_aTgrad_cusparse_settings(cusparseSpMMAlg_t alg) {
    internal::set_cusparse_spmm_settings(aTgrad_settings, CUDA_R_32I, &const_one, true, a, false, grad_wrapper,
                                         &const_zero, aTgrad_wrapper, alg);
}
template <>
void GCNConvOp<float>::init_aTgrad_cusparse_settings(cusparseSpMMAlg_t alg) {
    internal::set_cusparse_spmm_settings(aTgrad_settings, CUDA_R_32F, &const_one, true, a, false, grad_wrapper,
                                         &const_zero, aTgrad_wrapper, alg);
}
template <>
void GCNConvOp<double>::init_aTgrad_cusparse_settings(cusparseSpMMAlg_t alg) {
    internal::set_cusparse_spmm_settings(aTgrad_settings, CUDA_R_64F, &const_one, true, a, false, grad_wrapper,
                                         &const_zero, aTgrad_wrapper, alg);
}
#elif (CUDART_VERSION < 100100)

#endif
#endif

template <typename T>
void GCNConvOp<T>::init_eval_as_sparse(void) {
    this->output_tensor = new Tensor<T>(this->output_shape, this->mem_type);
    this->abc_tensor_slice = new Tensor<T>({n_vert_out, n_channel_out}, this->mem_type);
    switch (a->get_data_format()) {
        case SPARSEMATRIX_FORMAT_HOST_CSR:
            dense_format = SPARSEMATRIX_FORMAT_HOST_DENSE;
            this->b_wrapper = new spMatrix::hostSpMatrix_DENSE<T>(n_vert_in, n_channel_in);
            this->ab_wrapper = new spMatrix::hostSpMatrix_DENSE<T>(n_vert_out, n_channel_in);
            break;
#if defined(_HAS_CUDA_)
        case SPARSEMATRIX_FORMAT_CUSPARSE_CSR:
            dense_format = SPARSEMATRIX_FORMAT_CUSPARSE_DENSE;
            this->b_wrapper = new spMatrix::cusparseSpMatrix_DENSE<T>(n_vert_in, n_channel_in, this->mem_type);
            this->ab_wrapper = new spMatrix::cusparseSpMatrix_DENSE<T>(n_vert_out, n_channel_in, this->mem_type);
#if (CUDART_VERSION >= 100100)
            init_cusparse_settings(CUSPARSE_CSRMM_ALG1);
#endif
            break;
#endif
        default:
            std::fprintf(stderr, "Input sparse matrix format for GCNConvOp is not yet supported for initializing evaluation for as_sparse = true.\n");
            break;
    }
    this->b_tensor_slice = this->b_wrapper->get_data_ptr();
    this->ab_tensor_slice = this->ab_wrapper->get_data_ptr();
}

template <typename T>
void GCNConvOp<T>::init_eval_as_dense(void) {
    this->output_tensor = new Tensor<T>(this->output_shape, this->mem_type);
    dense_format = a->get_data_format();
    switch (dense_format) {
        case SPARSEMATRIX_FORMAT_HOST_DENSE:
            std::fprintf(stderr, "Input sparse matrix format for GCNConvOp is not yet supported for initializing evaluation for as_sparse = false.\n");
            break;
#if defined(_HAS_CUDA_)
        case SPARSEMATRIX_FORMAT_CUSPARSE_DENSE:
            cTbT_tensor = new Tensor<T>({n_samples, n_channel_out, n_vert_in}, this->mem_type);

            break;
#endif
        default:
            std::fprintf(stderr, "Input sparse matrix format for GCNConvOp is not yet supported for initializing evaluation for as_sparse = false.\n");
            break;
    }
}

template <typename T>
void GCNConvOp<T>::init_grad_as_sparse(void) {
    switch (dense_format) {
        case SPARSEMATRIX_FORMAT_HOST_DENSE:
            this->grad_wrapper = new spMatrix::hostSpMatrix_DENSE<T>(n_vert_out, n_channel_out);
            break;
#if defined(_HAS_CUDA_)
        case SPARSEMATRIX_FORMAT_CUSPARSE_DENSE:
            this->grad_wrapper = new spMatrix::cusparseSpMatrix_DENSE<T>(n_vert_out, n_channel_out, this->mem_type);
#if (CUDART_VERSION >= 1001000)
            init_aTgrad_cusparse_settings(CUSPARSE_CSRMM_ALG1);
#endif
            break;
#endif
        default:
            std::fprintf(stderr, "Input sparse matrix format for GCNConvOp is not yet supported for initializing grad computation for as_sparse = true.\n");
            break;
    }
    this->grad_tensor_slice = this->grad_wrapper->get_data_ptr();
}

template <typename T>
void GCNConvOp<T>::init_grad_as_dense(void) {
    switch (dense_format) {
        case SPARSEMATRIX_FORMAT_HOST_DENSE:
            std::fprintf(stderr, "Input sparse matrix format for GCNConvOp is not yet supported for initializing grad computation for as_sparse = false.\n");
            break;
#if defined(_HAS_CUDA_)
        case SPARSEMATRIX_FORMAT_CUSPARSE_DENSE:
            gTa_tensor = new Tensor<T>({n_samples, n_channel_out, n_vert_in}, this->mem_type);
            break;
#endif
        default:
            std::fprintf(stderr, "Input sparse matrix format for GCNConvOp is not yet supported for initializing grad computation for as_sparse = false.\n");
            break;
    }
}

template <typename T>
void GCNConvOp<T>::init_aTgrad_as_sparse(void) {
    switch (dense_format) {
        case SPARSEMATRIX_FORMAT_HOST_DENSE:
            this->aTgrad_wrapper = new spMatrix::hostSpMatrix_DENSE<T>(n_vert_in, n_channel_out);
            break;
#if defined(_HAS_CUDA_)
        case SPARSEMATRIX_FORMAT_CUSPARSE_DENSE:
            this->aTgrad_wrapper = new spMatrix::cusparseSpMatrix_DENSE<T>(n_vert_in, n_channel_out, this->mem_type);
#if (CUDART_VERSION >= 1001000)
            init_aTgrad_cusparse_settings(CUSPARSE_CSRMM_ALG1);
#endif
            break;
#endif
        default:
            std::fprintf(stderr, "Input sparse matrix format for GCNConvOp is not yet supported for initializing grad for as_sparse = true.\n");
            break;
    }
    this->aTgrad_tensor_slice = this->aTgrad_wrapper->get_data_ptr();
}

template <typename T>
void GCNConvOp<T>::init_bTaTg_as_dense(void) {
    switch (dense_format) {
        case SPARSEMATRIX_FORMAT_HOST_DENSE:
            std::fprintf(stderr, "Input sparse matrix format for GCNConvOp is not yet supported for initializing grad for as_sparse = false.\n");
            break;
#if defined(_HAS_CUDA_)
        case SPARSEMATRIX_FORMAT_CUSPARSE_DENSE:
            bTaTg_tensor = new Tensor<T>({n_samples, n_channel_in * n_channel_out}, this->mem_type);
            ones = new Tensor<T>({1, n_samples}, {ONE, {}}, this->mem_type);
            break;
#endif
        default:
            std::fprintf(stderr, "Input sparse matrix format for GCNConvOp is not yet supported for initializing grad for as_sparse = false.\n");
            break;
    }
}

template <typename T>
GCNConvOp<T>::GCNConvOp(spMatrix::sparseMatrix<T>* a, Operation<T>* b, Operation<T>* c, bool as_sparse, bool copy,
                        bool needs_grad)
    : Operation<T>({b, c}, needs_grad),
      as_sparse(as_sparse),
      a(a),
      b(b),
      c(c),
      copy(copy),
      //  just in case
      b_tensor(nullptr),
      b_tensor_slice(nullptr),
      b_wrapper(nullptr),
      c_tensor(nullptr),
      ab_tensor_slice(nullptr),
      ab_wrapper(nullptr),
      abc_tensor_slice(nullptr),
      grad_tensor_slice(nullptr),
      grad_wrapper(nullptr),
      aTgrad_tensor_slice(nullptr),
      aTgrad_wrapper(nullptr),
      aTgradcT_tensor_slice(nullptr),
      cTbT_tensor(nullptr),
      gTa_tensor(nullptr),
      bTaTg_tensor(nullptr),
      ones(nullptr),
      ab_settings(nullptr),
      aTgrad_settings(nullptr) {
    assert(a->get_memory_type() == b->get_memory_type());
    assert(OP_IS_SAME_MEMORY_TYPE(b, c));
    // assert(dynamic_cast<spMatrix::spMatrix_DENSE<T>*>(a) == nullptr &&
    //        "Sparse matrix for GCNConvOp must not be of dense format");  //  todo: slow to dynamic_cast, better method?
    assert(OP_IS_N_DIMENSIONAL(b, 3));
    assert(OP_IS_MATRIX(c));
    assert(a->get_shape(1) == b->get_output_shape(1));
    assert(b->get_output_shape(2) == c->get_output_shape(0));
    n_samples = b->get_output_shape(0);
    n_vert_in = a->get_shape(1);
    n_vert_out = a->get_shape(0);
    n_channel_in = c->get_output_shape(0);
    n_channel_out = c->get_output_shape(1);
    input_sample_size = n_vert_in * n_channel_in;
    output_sample_size = n_vert_out * n_channel_out;
    this->output_shape = {n_samples, n_vert_out, n_channel_out};
    this->mem_type = a->get_memory_type();
}

template <typename T>
GCNConvOp<T>::~GCNConvOp(void) {
    if (this->b_wrapper != nullptr) {  //  should also free b_tensor_slice
        delete this->b_wrapper;
    }
    if (this->ab_wrapper != nullptr) {  //  should also free ab_tensor_slice
        delete this->ab_wrapper;
    }
    if (this->abc_tensor_slice != nullptr) {
        delete this->abc_tensor_slice;
    }
    if (this->grad_wrapper != nullptr) {  //  should also free grad_tensor_slice
        delete this->grad_wrapper;
    }
    if (this->aTgrad_wrapper != nullptr) {  //  should also free aTgrad_wrapper
        delete this->aTgrad_wrapper;
    }
    if (this->cTbT_tensor != nullptr) {
        delete this->cTbT_tensor;
    }
    if (this->gTa_tensor != nullptr) {
        delete this->gTa_tensor;
    }
    if (this->bTaTg_tensor != nullptr) {
        delete this->bTaTg_tensor;
    }
    if (this->ones != nullptr) {
        delete this->ones;
    }
    if (this->aTgradcT_tensor_slice != nullptr) {
        delete this->aTgradcT_tensor_slice;
    }
    if (this->ab_settings != nullptr) {
#if defined(_HAS_CUDA_)
#if (CUDART_VERSION >= 100100)
        //  should only be set for version 10.1+
        if (this->dense_format == SPARSEMATRIX_FORMAT_CUSPARSE_DENSE) {
            cudaErrchk(cudaFree(AS_TYPE(math::spgemm_cusparse_settings*, this->ab_settings)->workspace));
            delete AS_TYPE(math::spgemm_cusparse_settings*, this->ab_settings);
        }
#endif
#endif
    }
    if (this->aTgrad_settings != nullptr) {
#if defined(_HAS_CUDA_)
#if (CUDART_VERSION >= 100100)
        //  should only be set for version 10.1+
        if (this->dense_format == SPARSEMATRIX_FORMAT_CUSPARSE_DENSE) {
            cudaErrchk(cudaFree(AS_TYPE(math::spgemm_cusparse_settings*, this->aTgrad_settings)->workspace));
            delete AS_TYPE(math::spgemm_cusparse_settings*, this->aTgrad_settings);
        }
#endif
#endif
    }
}

template <typename T>
Tensor<T>* GCNConvOp<T>::_eval(bool recompute) {
    b_tensor = b->eval(recompute);
    c_tensor = c->eval(recompute);
    if (as_sparse) {
        if (b_tensor_slice == nullptr) {
            init_eval_as_sparse();
        }
        for (unsigned sample_idx = 0; sample_idx < n_samples; ++sample_idx) {  //  for each sample
            b_tensor_slice->copy_from(*b_tensor, sample_idx * input_sample_size, input_sample_size);
            // math::spgematmul(const_one, false, a, false, b_wrapper, const_zero, ab_wrapper, ab_settings, false);
            // math::matmul(const_one, false, ab_tensor_slice, false, c_tensor, const_zero, abc_tensor_slice);
            // evil hack to avoid transposing every round of evaluation, better method?
            math::spgematmul(const_one, false, a, false, b_wrapper, const_zero, ab_wrapper, ab_settings, true);
            ab_tensor_slice->reshape({n_channel_in, n_vert_out});
            math::matmul(const_one, true, ab_tensor_slice, false, c_tensor, const_zero, abc_tensor_slice);
            ab_tensor_slice->reshape({n_vert_out, n_channel_in});
            // evil hack ends here
            this->output_tensor->copy_from(*this->abc_tensor_slice, 0, output_sample_size,
                                           sample_idx * output_sample_size);
        }
    } else {
        if (cTbT_tensor == nullptr) {
            init_eval_as_dense();
        }
#if defined(_HAS_CUDA_)
        //  compute c_tensor^T * b_tensor_slice^T, stored in cTbT_tensor in column-major (so in row-major it is b_slice * c)
        this->cublasGemmStridedBatchedWrap(false, false, n_channel_out, n_vert_in, n_channel_in, const_one,
                                           c_tensor->get_ptr(), n_channel_out, 0, b_tensor->get_ptr(), n_channel_in,
                                           input_sample_size, const_zero, cTbT_tensor->get_ptr(), n_channel_out,
                                           n_vert_in * n_channel_out, n_samples);
        //  compute cTbT_tensor * a^T, stored in out in column-major (so in row-major it is a * b_slice * c)
        this->cublasGemmStridedBatchedWrap(
            false, false, n_channel_out, n_vert_out, n_vert_in, const_one, cTbT_tensor->get_ptr(), n_channel_out,
            n_channel_out * n_vert_in, AS_TYPE(spMatrix::spMatrix_DENSE<T>*, a)->get_data_ptr()->get_ptr(), n_vert_in,
            0, const_zero, this->output_tensor->get_ptr(), n_channel_out, output_sample_size, n_samples);
#endif
    }
    return this->output_tensor;
}

template <typename T>
Tensor<T>* GCNConvOp<T>::_grad(Operation<T>* consumer, Operation<T>* var, Tensor<T>* grad) {
    assert(T_IS_N_DIMENSIONAL(grad, 3));
    assert(grad->get_shape(0) == n_samples);
    assert(grad->get_shape(1) == n_vert_out);
    assert(grad->get_shape(2) == n_channel_out);
    // assert(grad->get_shape() == std::vector<unsigned>({n_samples, n_vert_out, n_channel_out}));
    Tensor<T>* out = this->_grad_cache[(uintptr_t) var];
    if (as_sparse && grad_wrapper == nullptr) {
        init_grad_as_sparse();
    }
    if (var == b) {
        //  out_{i} = a^T * grad_{i} * c^T
        //  compute a^T * grad_{i}, put in out
        c_tensor = c->eval(false);
        if (out == NULL) {
            out = new Tensor<T>(b->get_output_shape(), {NONE, {}}, this->mem_type);
            this->_grad_cache[(uintptr_t) b] = out;
        }
        if (as_sparse) {
            if (aTgrad_wrapper == nullptr) {
                init_aTgrad_as_sparse();
            }
            for (unsigned sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
                this->grad_tensor_slice->copy_from(*grad, sample_idx * output_sample_size, output_sample_size);
                /*  math::spgematmul(const_one, true, a, false, this->grad_wrapper, const_zero, aTgrad_wrapper,
                 *aTgrad_settings, false); math::matmul(const_one, true, aTgrad_tensor_slice, true, c_tensor,
                 *const_zero, this->aTgradcT_tensor_slice);
                 **///  evil hack to avoid transposing every round of evaluation
                math::spgematmul(const_one, true, a, false, this->grad_wrapper, const_zero, aTgrad_wrapper,
                                 aTgrad_settings, true);
                aTgrad_tensor_slice->reshape({n_channel_out, n_vert_in});
                math::matmul(const_one, true, aTgrad_tensor_slice, true, c_tensor, const_zero,
                             this->aTgradcT_tensor_slice);
                aTgrad_tensor_slice->reshape({n_vert_in, n_channel_out});
                //  evil hack ends here
                out->copy_from(*this->aTgradcT_tensor_slice, 0, input_sample_size, sample_idx * input_sample_size);
            }
        } else {
#if defined(_HAS_CUDA_)
            if (gTa_tensor == nullptr) {
                init_grad_as_dense();
            }
            //  compute grad^T * (a^T)^T , stored in gTa_tensor in column-major (so in row-major it is a^T * grad)
            this->cublasGemmStridedBatchedWrap(
                false, true, n_channel_out, n_vert_in, n_vert_out, const_one, grad->get_ptr(), n_channel_out,
                output_sample_size, AS_TYPE(spMatrix::spMatrix_DENSE<T>*, a)->get_data_ptr()->get_ptr(), n_vert_in, 0,
                const_zero, gTa_tensor->get_ptr(), n_channel_out, n_channel_out * n_vert_in, n_samples);
            //  compute (c^T)^T * gTa , stored in out in column-major (so in row-major it is a^T * grad * c^T)
            this->cublasGemmStridedBatchedWrap(true, false, n_channel_in, n_vert_in, n_channel_out, const_one,
                                               c_tensor->get_ptr(), n_channel_out, 0, gTa_tensor->get_ptr(),
                                               n_channel_out, n_channel_out * n_vert_in, const_zero, out->get_ptr(),
                                               n_channel_in, input_sample_size, n_samples);
#endif
        }

    } else {  // if (var == c)
        //  out = \sum_{i}( (a * b_{i})^T * grad_{i} )
        //  recompute a*b_{i} with spgemm, matmul with grad_{i}
        //  reuse wrapper and settings from computing spgemm(a, b_{i})
        b_tensor = b->eval(false);
        if (out == NULL) {
            out = new Tensor<T>(c->get_output_shape(), {NONE, {}}, this->mem_type);
            this->_grad_cache[(uintptr_t) c] = out;
        }
        if (as_sparse) {
            for (unsigned sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
                this->b_tensor_slice->copy_from(*b_tensor, sample_idx * input_sample_size, input_sample_size);
                this->grad_tensor_slice->copy_from(*grad, sample_idx * output_sample_size, output_sample_size);
                /*  math::spgematmul(const_one, false, a, false, b_wrapper, const_zero, ab_wrapper, ab_settings, false);
                 *  math::matmul(const_one, true, ab_tensor_slice, false, grad_tensor_slice, (sample_idx == 0) ?
                 *const_zero : const_one, out);  //  reset out / accumulate
                 **///  evil hack to avoid transposing every round of evaluation
                math::spgematmul(const_one, false, a, false, b_wrapper, const_zero, ab_wrapper, ab_settings, true);
                ab_tensor_slice->reshape({n_channel_in, n_vert_out});
                math::matmul(const_one, false, ab_tensor_slice, false, grad_tensor_slice,
                             (sample_idx == 0) ? const_zero : const_one, out);
                ab_tensor_slice->reshape({n_vert_out, n_channel_in});
                //  evil hack ends here
            }
        } else {
#if defined(_HAS_CUDA_)
            if (gTa_tensor == nullptr) {
                init_grad_as_dense();
            }
            if (bTaTg_tensor == nullptr) {
                init_bTaTg_as_dense();
            }
            //  compute grad^T * (a^T)^T , stored in gTa_tensor in column-major (so in row-major it is a^T * grad)
            this->cublasGemmStridedBatchedWrap(
                false, true, n_channel_out, n_vert_in, n_vert_out, const_one, grad->get_ptr(), n_channel_out,
                output_sample_size, AS_TYPE(spMatrix::spMatrix_DENSE<T>*, a)->get_data_ptr()->get_ptr(), n_vert_in, 0,
                const_zero, gTa_tensor->get_ptr(), n_channel_out, n_channel_out * n_vert_in, n_samples);
            //  compute gTa * (b_slice^T)^T , stored in bTaTg_tensor in column-major (so in row-major it is b^T * a^T * grad)
            this->cublasGemmStridedBatchedWrap(
                false, true, n_channel_out, n_channel_in, n_vert_in, const_one, gTa_tensor->get_ptr(), n_channel_out,
                n_channel_out * n_vert_in, b_tensor->get_ptr(), n_channel_in, input_sample_size, const_zero,
                bTaTg_tensor->get_ptr(), n_channel_out, n_channel_in * n_channel_out, n_samples);
            //  reduced with respect to the sample axis (0-th)
            math::matmul<T>(const_one, false, ones, false, bTaTg_tensor, const_zero, out);
#endif
        }
    }
    return out;
}

template class GCNConvOp<int>;
template class GCNConvOp<float>;
template class GCNConvOp<double>;

template <typename T>
GCNConvOp<T>* gcnconv(spMatrix::sparseMatrix<T>* a, Operation<T>* b, Operation<T>* c, bool as_sparse, bool copy,
                      bool needs_grad) {
    return new GCNConvOp<T>(a, b, c, as_sparse, copy, needs_grad);
}
template GCNConvOp<int>* gcnconv(spMatrix::sparseMatrix<int>*, Operation<int>*, Operation<int>*, bool, bool, bool);
template GCNConvOp<float>* gcnconv(spMatrix::sparseMatrix<float>*, Operation<float>*, Operation<float>*, bool, bool,
                                   bool);
template GCNConvOp<double>* gcnconv(spMatrix::sparseMatrix<double>*, Operation<double>*, Operation<double>*, bool, bool,
                                    bool);

}  // namespace op
}  // namespace magmadnn
