#pragma once

#include <string>
#include "compute/operation.h"
#include "math/spgematmul.h"
#include "sparseMatrix/sparseMatrix.h"
#include "tensor/tensor.h"

namespace magmadnn {
namespace op {

//  Operation to compute alpha*a*b where alpha is scalar, a is sparse matrix, b is an operation
//  use a for reference (e.g. constant), only propagate b
template <typename T>
class SpgemmOp : public Operation<T> {
   public:
    SpgemmOp(T alpha, spMatrix::sparseMatrix<T>* a, Operation<T>* b, bool copy = true, bool needs_grad = true);
    ~SpgemmOp(void);
    inline std::string to_string(void) {
        return "SPGEMM( " + (alpha == (T) 1 ? "" : (std::to_string(alpha)) + " , ") + a->to_string() + " , " +
               b->to_string() + " )";
    }

   protected:
    bool copy;
    T alpha;
    const T beta = (T) 0;
    spMatrix::sparseMatrix<T>* a;
    Operation<T>* b;
    Tensor<T>* b_tensor;
    //  struct wrapper used for calling spgemm routines
    //  todo: there may be a better way to do this? (e.g. massive use of templates)
    spMatrix::spMatrix_DENSE<T>* b_wrapper;  //  wrapper for b
    spMatrix::spMatrix_DENSE<T>* ab_wrapper;  //  wrapper for ab
    spMatrix::spMatrix_DENSE<T>* grad_wrapper;  //  wrapper for grad
    spMatrix::spMatrix_DENSE<T>* out_wrapper;  //  wrapper for out
    //  store descriptor to save time in casting
    void* a_descriptor;
    void* b_descriptor;
    void* ab_descriptor;
    void* grad_descriptor;
    void* out_desctiptor;

    spMatrix_format sp_mat_format;  //  format for a
    void* settings;  //  settings for calling spgemm routines
    void* grad_settings;  //  settings for computing grad

    Tensor<T>* _eval(bool recompute);
    Tensor<T>* _grad(Operation<T>* consumer, Operation<T>* var, Tensor<T>* grad);

   private:
    void init(void);
    void init_grad(Tensor<T>* grad, Tensor<T>* out);
#if defined(_HAS_CUDA_)
    void init_cusparse_csr(void);
    void init_grad_cusparse_csr(Tensor<T>* grad, Tensor<T>* out);
#endif
};

template <typename T>
SpgemmOp<T>* spgemm(spMatrix::spMatrix_DENSE<T>* a, Operation<T>* b, bool copy = true, bool needs_grad = true);

}  //  namespace op
}  //  namespace magmadnn
