#include "compute/gcnconv/gcnconvop.h"

namespace magmadnn {
namespace op {

#if defined(_HAS_CUDA_)

#endif

template <typename T>
void GCNConvOp<T>::init(void){
    this->b_tensor = new Tensor<T>(b->get_output_shape(), this->mem_type);
    this->b_tensor_slice = new Tensor<T>({b->get_output_shape(1), b->get_output_shape(2)}, this->mem_type);
    this->c_tensor = new Tensor<T>(c->get_output_shape(), this->mem_type);
    this->ab_tensor_slice = new Tensor<T>({a->get_shape(0), b->get_output_shape(2)}, this->mem_type);
    switch (a->get_data_format())
    {
    case SPARSEMATRIX_FORMAT_HOST_CSR:
        dense_format = SPARSEMATRIX_FORMAT_HOST_DENSE;
        break;
#if defined(_HAS_CUDA_)
    case SPARSEMATRIX_FORMAT_CUSPARSE_CSR:
        dense_format = SPARSEMATRIX_FORMAT_CUSPARSE_DENSE;
        this->b_wrapper = new spMatrix::cusparseSpMatrix_DENSE<T>(this->b_tensor_slice, this->mem_type, false);
        this->ab_wrapper = new spMatrix::cusparseSpMatrix_DENSE<T>(this->ab_tensor_slice, this->mem_type, false);
        break;
#endif
    default:
        std::fprintf(stderr, "Input sparse matrix format for GCNConvOp is not supported.\n");
        break;
    }
}

template <typename T>
GCNConvOp<T>::GCNConvOp(spMatrix::sparseMatrix<T>* a, Operation<T>* b, Operation<T>* c, bool copy = true, bool needs_grad = true): Operation<T>({b, c}, needs_grad), a(a), b(b), c(c), copy(copy), b_tensor_slice(nullptr), c_tensor(nullptr), ab_tensor_slice(nullptr), b_wrapper(nullptr), ab_wrapper(nullptr){
    assert(a->get_memory_type() == b->get_memory_type());
    assert(OP_IS_SAME_MEMORY_TYPE(b, c));
    assert(dynamic_cast<spMatrix::spMatrix_DENSE<T>*>(a) == nullptr &&
           "Sparse matrix for GCNConvOp must not be of dense format");
    assert(OP_IS_N_DIMENSIONAL(b, 3));
    assert(OP_IS_MATRIX(c));
    assert(a->get_shape(1) == b->get_output_shape(1));
    assert(b->get_output_shape(2) == c->get_output_shape(0));
    n_samples = b->get_output_shape(0);
    sample_size = b->get_output_shape(1) * b->get_output_shape(2);
    this->output_shape = {n_samples, a->get_shape(0), c->get_output_shape(1)};
    this->mem_type = a->get_memory_type();
    this->output_tensor = Tensor<T>(this->output_shape, this->mem_type);
    init();
}

template <typename T>
GCNConvOp<T>::~GCNConvOp(void){
    if (this->b_tensor_slice != nullptr){
        delete this->b_tensor_slice;
    }
    if (this->c_tensor != nullptr){
        delete this->c_tensor;
    }
    if (this->ab_tensor_slice!=nullptr){
        delete this->ab_tensor_slice;
    }
    if (this->b_wrapper != nullptr){
        delete this->b_wrapper;
    }
    if (this->ab_wrapper != nullptr){
        delete this->ab_wrapper;
    }
    
}

template <typename T>
Tensor<T>* GCNConvOp<T>::_eval(bool recompute){
    b_tensor = b->_eval(recompute);
    c_tensor = c->_eval(recompute);
    for (unsigned sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
        this->b_tensor_slice->copy_from(*b_tensor, sample_idx * sample_size, sample_size);
        b_wrapper->set_mat(b_tensor_slice);
        math::spgematmul(alpha, false, a, false, b_wrapper, beta, ab_wrapper, )
    }
}

template  <typename T>
Tensor<T>* GCNConvOp<T>::_grad(Operation<T>* consumer, Operation<T>* var, Tensor<T>* grad){

}

template class GCNConvOp<int>;
template class GCNConvOp<float>;
template class GCNConvOp<double>;

template <typename T>
GCNConvOp<T>* gcnconv(spMatrix::sparseMatrix<T>* a, Operation<T>* b, Operation<T>* c, bool copy = true, bool needs_grad = true) {
    return new GcnConvOp<T>(a, b, c, copy, needs_grad);
}

template GCNConvOp<int>* gcnconv(spMatrix::sparseMatrix<int>*, Operation<int>*, Operation<int>*, bool, bool);
template GCNConvOp<float>* gcnconv(spMatrix::sparseMatrix<float>*, Operation<float>*, Operation<float>*, bool, bool);
template GCNConvOp<double>* gcnconv(spMatrix::sparseMatrix<double>*, Operation<double>*, Operation<double>*, bool, bool);

}  // namespace op
}  // namespace magmadnn
