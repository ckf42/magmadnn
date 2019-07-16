#pragma once

#include <string>
#include "compute/operation.h"
#include "math/spgematmul.h"
#include "sparseMatrix/sparseMatrix.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace op {

//  takes two parameters: sparse a, Tensor b
//  use a for reference (e.g. constant), only propagate b
template <typename T>
class SpgemmOp : public Operation<T> {
   public:
    SpgemmOp(T alpha, spMatrix::sparseMatrix<T>* a, Operation<T>* b, T beta, Operation<T>* c, bool copy = true,bool needs_grad = true);
    std::string to_string(void) { return "SPGEMM(" + a->to_string() + " * " + b->to_string() + ")"; }
    ~SpgemmOp(void);

   protected:
    bool copy;
    T alpha;
    T beta;
    spMatrix::sparseMatrix<T>* a;
    Operation<T>* b;
    Operation<T>* c;
    Tensor<T>* b_tensor;
    Tensor<T>* c_tensor;
    //  struct wrapper used for calling spgemm routines
    void* a_descriptor;  //  descriptor for a
    void* b_wrapper;  //  wrapper for b
    void* b_descriptor;  //  descriptor for b
    void* c_wrapper;  //  wrapper for c
    void* c_descriptor;  //  descriptor for c

    spMatrix_format sp_mat_format;  //  format for a
    void* settings;  //  settings for calling spgemm routines

#if defined(_HAS_CUDA_)

#endif

    Tensor<T>* _eval(bool recompute);
    Tensor<T>* _grad(Operation<T>* consumer, Operation<T>* var, Tensor<T>* grad);

   private:
    void init_desc(void);
    void init_host_desc(void);
#if defined(_HAS_CUDA_)
    void init_cusparse_desc(void);
#endif
};

// template <typename T>
// SpgemmOp<T> *spgemm(){}

}  //  namespace op
}  //  namespace magmadnn
