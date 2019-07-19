#pragma once

#include <string>
#include "compute/operation.h"
#include "math/spgematmul.h"
#include "sparseMatrix/sparseMatrix.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace op {

//  Operation to compute x_{ijk} = a_{jm} * b_{imn} * c_{nk}, where a is 2D sparse matrix, b is an operation with 3D output, c is an operation with 2D output
//  use a for reference (e.g. constant), only propagate b
//  at the moment, multiplication is processed sample by sample
//  todo: use better routine?
template <typename T>
class GCNConvOp : public Operation<T> {
   public:
    GCNConvOp(spMatrix::sparseMatrix<T>* a, Operation<T>* b, Operation<T>* c, bool copy = true, bool needs_grad = true);
    ~GCNConvOp(void);
    inline std::string to_string(void) {
        return "GCNCONV( " + a->to_string() + " , " + b->to_string() + " , " + c->to_string() + " )";
    }

   protected:
    const T alpha = (T) 1;
    const T beta = (T) 0;
    unsigned n_samples;
    unsigned sample_size;
    spMatrix::sparseMatrix<T>* a;
    Operation<T>* b;
    Operation<T>* c;
    bool copy;
    Tensor<T>* b_tensor;
    Tensor<T>* b_tensor_slice;  //  one sample
    Tensor<T>* c_tensor;
    Tensor<T>* ab_tensor_slice;  //  stores the result of spgemm(a, b_tensor_slice)
    //  wrapper for calling spgemm
    spMatrix_format dense_format;
    spMatrix::spMatrix_DENSE<T>* b_wrapper;
    spMatrix::spMatrix_DENSE<T>* ab_wrapper;
    void* settings;

    Tensor<T>* _eval(bool recompute);
    Tensor<T>* _grad(Operation<T>* consumer, Operation<T>* var, Tensor<T>* grad);

   private:
    void init(void);
    void init_grad(void);
#if defined(_HAS_CUDA_)
    void 
#endif


};

template <typename T>
GcnConvOp<T>* gcnconv(spMatrix::sparseMatrix<T>* a, Operation<T>* b,, Operation<T>* c, bool copy = true, bool needs_grad = true);

}  //  namespace op
}  //  namespace magmadnn
