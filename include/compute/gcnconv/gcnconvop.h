#pragma once

#include <string>
#include "compute/gcnconv/gcnconvop_internal.h"
#include "compute/operation.h"
#include "compute/tensor_operations.h"
#include "math/spgematmul.h"
#include "sparseMatrix/sparseMatrix.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace op {

//  Operation to compute x_{ijk} = a_{jm} * b_{imn} * c_{nk}, where a is 2D sparse matrix, b is an operation with 3D
//  output, c is an operation with 2D output
//  use a as constant, only propagate b, c
//  input shape: a { n_vert_out, n_vert_in }
//               b { n_samples, n_vert_in, n_channel_in }
//               c { n_channel_in, n_channel_out }
//  output shape: x { n_samples, n_vert_out, n_channel_out }
//  at the moment, multiplication is processed sample by sample
//  todo: use better routine to process in batch / in parallel ?
//  todo: reduce number of wrapper used
template <typename T>
class GCNConvOp : public Operation<T> {
   public:
    GCNConvOp(spMatrix::sparseMatrix<T>* a, Operation<T>* b, Operation<T>* c, bool copy = true, bool needs_grad = true);
    ~GCNConvOp(void);
    inline std::string to_string(void) {
        return "GCNCONV( " + a->to_string() + " * " + b->to_string() + " * " + c->to_string() + " )";
    }

   private:
    const T const_one = (T) 1;
    const T const_zero = (T) 0;
    inline void init_eval(void);
    inline void init_grad(void);
    inline void init_aTgrad(void);
#if defined(_HAS_CUDA_)
    inline void init_cusparse_settings(cusparseSpMMAlg_t alg);
    inline void init_aTgrad_cusparse_settings(cusparseSpMMAlg_t alg);
#endif

   protected:
    unsigned n_samples;
    unsigned n_vert_in, n_vert_out;
    unsigned n_channel_in, n_channel_out;
    unsigned input_sample_size;
    unsigned output_sample_size;
    spMatrix::sparseMatrix<T>* a;
    Operation<T>* b;
    Operation<T>* c;
    bool copy;
    //  wrapper for calling spgemm
    Tensor<T>* b_tensor;
    Tensor<T>* b_tensor_slice;               //  one sample
    spMatrix::spMatrix_DENSE<T>* b_wrapper;  //  sparse matrix object connected to b_tensor_slice
    Tensor<T>* c_tensor;
    Tensor<T>* ab_tensor_slice;                   //  stores the result of spgemm(a, b_tensor_slice)
    spMatrix::spMatrix_DENSE<T>* ab_wrapper;      //  sparse matrix object connected to ab_tensor_slice
    Tensor<T>* abc_tensor_slice;                  //  stores the result of matmul(ab_tensor_slice, c_tensor)
    Tensor<T>* grad_tensor_slice;                 //  one sample
    spMatrix::spMatrix_DENSE<T>* grad_wrapper;    //  sparse matrix object connected to grad_tensor_slice
    Tensor<T>* aTgrad_tensor_slice;               //  stores the results of spgemm(a^T, grad_tensor_slice)
    spMatrix::spMatrix_DENSE<T>* aTgrad_wrapper;  //  sparse matrix object connected to aTgrad_tensor_slice
    Tensor<T>* aTgradcT_tensor_slice;             //  stores the result of matmul(aTgrad_tensor_slice, c_tensor)

    spMatrix_format dense_format;  //  format to store dense matrices for spgemm
    void* ab_settings;
    void* aTgrad_settings;

    Tensor<T>* _eval(bool recompute);
    Tensor<T>* _grad(Operation<T>* consumer, Operation<T>* var, Tensor<T>* grad);
};

template <typename T>
GCNConvOp<T>* gcnconv(spMatrix::sparseMatrix<T>* a, Operation<T>* b, Operation<T>* c, bool copy = true,
                      bool needs_grad = true);

}  //  namespace op
}  //  namespace magmadnn
