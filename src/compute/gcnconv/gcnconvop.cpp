#include "compute/gcnconv/gcnconvop.h"

namespace magmadnn {
namespace op {

#if defined(_HAS_CUDA_)
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
void GCNConvOp<float>::init_aTgrad_cusparse_settings(cusparseSpMMAlg_t alg) {
    internal::set_cusparse_spmm_settings(aTgrad_settings, CUDA_R_32F, &const_one, true, a, false, grad_wrapper,
                                         &const_zero, aTgrad_wrapper, alg);
}
void GCNConvOp<double>::init_aTgrad_cusparse_settings(cusparseSpMMAlg_t alg) {
    internal::set_cusparse_spmm_settings(aTgrad_settings, CUDA_R_64F, &const_one, true, a, false, grad_wrapper,
                                         &const_zero, aTgrad_wrapper, alg);
}
#endif

template <typename T>
void GCNConvOp<T>::init_eval(void) {
    this->output_tensor = Tensor<T>(this->output_shape, this->mem_type);
    this->abc_tensor_slice = new Tensor<T>({n_vert_in, n_channel_in}, this->mem_type);
    switch (a->get_data_format()) {
        case SPARSEMATRIX_FORMAT_HOST_CSR:
            dense_format = SPARSEMATRIX_FORMAT_HOST_DENSE;
            this->b_wrapper = new spMatrix::hostSpMatrix_DENSE<T>(n_vert_in, n_channel_in, false);
            this->ab_wrapper = new spMatrix::hostSpMatrix_DENSE<T>(n_vert_out, n_channel_in, false);
            break;
#if defined(_HAS_CUDA_)
        case SPARSEMATRIX_FORMAT_CUSPARSE_CSR:
            dense_format = SPARSEMATRIX_FORMAT_CUSPARSE_DENSE;
            this->b_wrapper = new spMatrix::cusparseSpMatrix_DENSE<T>(n_vert_in, n_channel_in, this->mem_type, false);
            this->ab_wrapper = new spMatrix::cusparseSpMatrix_DENSE<T>(n_vert_out, n_channel_in, this->mem_type, false);
                                init_cusparse_settings((CUSPARSE_CSRMM_ALG1);
				break;
#endif
			default:
				std::fprintf(stderr, "Input sparse matrix format for GCNConvOp is not yet supported.\n");
				break;
    }
    this->b_tensor_slice = this->b_wrapper->get_data_ptr();
    this->ab_tensor_slice = this->ab_wrapper->get_data_ptr();
}

template <typename T>
void GCNConvOp<T>::init_grad(void) {
    switch (dense_format) {
        case SPARSEMATRIX_FORMAT_HOST_DENSE:
            this->grad_wrapper = new spMatrix::hostSpMatrix_DENSE<T>(n_vert_out, n_channel_out, false);
            break;
#if defined(_HAS_CUDA_)
        case SPARSEMATRIX_FORMAT_CUSPARSE_DENSE:
            this->grad_wrapper = new spMatrix::cusparseSpMatrix_DENSE<T>(
                n_vert_out, n_channel_out, this->mem_type, false) init_aTgrad_cusparse_settings(CUSPARSE_CSRMM_ALG1);
            break;
#endif
        default:
            std::fprintf(stderr, "Input sparse matrix format for GCNConvOp is not yet supported.\n");
            break;
    }
    this->grad_tensor_slice = this->grad_wrapper->get_data_ptr();
}

template <typename T>
void GCNConvOp<T>::init_aTgrad(void) {
    switch (dense_format) {
        case SPARSEMATRIX_FORMAT_HOST_DENSE:
            this->aTgrad_wrapper = new spMatrix::hostSpMatrix_DENSE<T>(n_vert_in, n_channel_out, false);
            break;
#if defined(_HAS_CUDA_)
        case SPARSEMATRIX_FORMAT_CUSPARSE_DENSE:
            this->aTgrad_wrapper =
                new spMatrix::cusparseSpMatrix_DENSE<T>(n_vert_in, n_channel_out, this->mem_type, false);
            init_aTgrad_cusparse_settings(CUSPARSE_CSRMM_ALG1);
            break;
#endif
        default:
            std::fprintf(stderr, "Input sparse matrix format for GCNConvOp is not yet supported.\n");
            break;
    }
    this->aTgrad_tensor_slice = this->aTgrad_wrapper->get_data_ptr();
}

template <typename T>
GCNConvOp<T>::GCNConvOp(const spMatrix::sparseMatrix<T>* a, Operation<T>* b, Operation<T>* c, bool copy = true,
                        bool needs_grad = true)
    : Operation<T>({b, c}, needs_grad),
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
      ab_settings(nullptr),
      aTgrad_settings(nullptr) {
    assert(a->get_memory_type() == b->get_memory_type());
    assert(OP_IS_SAME_MEMORY_TYPE(b, c));
    assert(dynamic_cast<spMatrix::spMatrix_DENSE<T>*>(a) == nullptr &&
           "Sparse matrix for GCNConvOp must not be of dense format");  //  todo: slow to dynamic_cast, better method?
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
    if (this->aTgradcT_tensor_slice != nullptr) {
        delete this->aTgradcT_tensor_slice;
    }
    if (this->ab_settings != nullptr) {
#if defined(_HAS_CUDA_)
        if (this->dense_format == SPARSEMATRIX_FORMAT_CUSPARSE_DENSE) {
            cudaErrchk(cudaFree(AS_TYPE(math::spgemm_cusparse_settings*, this->ab_settings)->workspace));
        }
#endif
        delete this->ab_settings;
    }
    if (this->aTgrad_settings != nullptr) {
#if defined(_HAS_CUDA_)
        if (this->dense_format == SPARSEMATRIX_FORMAT_CUSPARSE_DENSE) {
            cudaErrchk(cudaFree(AS_TYPE(math::spgemm_cusparse_settings*, this->aTgrad_settings)->workspace));
        }
#endif
        delete this->aTgrad_settings;
    }
}

template <typename T>
Tensor<T>* GCNConvOp<T>::_eval(bool recompute) {
    if (b_tensor_slice == nullptr) {
        init_eval();
    }
    b_tensor = b->_eval(recompute);
    c_tensor = c->_eval(recompute);
    for (unsigned sample_idx = 0; sample_idx < n_samples; ++sample_idx) {  //  for each sample
        b_tensor_slice->copy_from(*b_tensor, sample_idx * input_sample_size, input_sample_size);
        // b_wrapper->set_mat(b_tensor_slice);
        math::spgematmul(const_one, false, a, false, b_wrapper, const_zero, ab_wrapper, ab_settings);
        // ab_wrapper->get_uncompressed_mat(ab_tensor_slice);
        math::matmul(const_one, false, ab_tensor_slice, false, c_tensor, const_zero, abc_tensor_slice);
        this->output_tensor->copy_from(*abc_tensor_slices, 0, output_sample_size, sample_idx * output_sample_size);
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
    if (grad_wrapper == nullptr) {
        init_grad();
    }
    if (var == b) {
        //  out_{i} = a^T * grad_{i} * c^T
        //  compute a^T * grad_{i}, put in out
        c_tensor = c->eval(recompute);
        if (out == NULL) {
            out = new Tensor<T>(b->get_output_shape(), {NONE, {}}, this->mem_type);
            this->_grad_cache[(uintptr_t) b] = out;
        }
        if (aTgrad_wrapper == nullptr) {
            init_aTgrad();
        }
        for (unsigned sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
            this->grad_tensor_slice->copy_from(*grad, sample_idx * output_sample_size, output_sample_size);
            math::spgematmul(const_one, true, a, false, this->grad_wrapper, const_zero, aTgrad_wrapper,
                             aTgrad_settings);
            math::matmul(const_one, false, aTgrad_tensor_slice, true, c_tensor, const_zero,
                         this->aTgradcT_tensor_slice);
            out->copy_from(*this->aTgradcT_tensor_slice, 0, input_sample_size, sample_idx * input_sample_size);
        }
    } else {  // if (var == c)
        //  out = \sum_{i}( (a * b_{i})^T * grad_{i} )
        //  recompute a*b_{i} with spgemm, matmul with grad_{i}
        //  reuse wrapper and settings from computing spgemm(a, b_{i})
        b_tensor = b->eval(recompute);
        if (out == NULL) {
            out = new Tensor<T>(c->get_output_shape(), {NONE, {}}, this->mem_type);
            this->_grad_cache[(uintptr_t) c] = out;
        }
        for (unsigned sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
            this->b_tensor_slice->copy_from(*b_tensor, sample_idx * input_sample_size, input_sample_size);
            math::spgematmul(const_one, false, a, false, b_wrapper, const_zero, ab_wrapper, ab_settings);
            this->grad_tensor_slice->copy_from(*grad, sample_idx * output_sample_size, output_sample_size);
            math::matmul(const_one, true, ab_tensor_slice, false, grad_tensor_slice,
                         (sample_idx == 0) ? const_zero : const_one, out);  //  reset out / accumulate
        }
    }
    return out;
}

template class GCNConvOp<int>;
template class GCNConvOp<float>;
template class GCNConvOp<double>;

template <typename T>
GCNConvOp<T>* gcnconv(spMatrix::sparseMatrix<T>* a, Operation<T>* b, Operation<T>* c, bool copy = true,
                      bool needs_grad = true) {
    return new GCNConvOp<T>(a, b, c, copy, needs_grad);
}
template GCNConvOp<int>* gcnconv(spMatrix::sparseMatrix<int>*, Operation<int>*, Operation<int>*, bool, bool);
template GCNConvOp<float>* gcnconv(spMatrix::sparseMatrix<float>*, Operation<float>*, Operation<float>*, bool, bool);
template GCNConvOp<double>* gcnconv(spMatrix::sparseMatrix<double>*, Operation<double>*, Operation<double>*, bool,
                                    bool);

}  // namespace op
}  // namespace magmadnn
