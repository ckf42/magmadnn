#pragma once

#include "compute/operation.h"
#include "tensor/tensor.h"
#include "sparseMatrix/sparseMatrix.h"
#include "math/spgematmul.h"
#include <string>

namespace magmadnn{
namespace op{

//  takes two parameters: sparse a, Tensor b
//  use a for reference, only propagate b
template <typename T>
class spgemmOp: public Operation<T>{
    public:
    spgemmOp(T alpha, spMatrix::sparseMatrix<T>* a, Operation<T>* b, T beta, Operation<T> *c, bool copy = true, bool needs_grad = true);
    std::string to_string(void){return "SPGEMM(" + ????? + " * " + b->to_string() + ")" ;}
    ~spgemmOp(void);
    protected:
    T alpha;
    T beta;
    spMatrix::sparseMatrix<T>* a;
    Operation<T>* b;
    Operation<T>* c;
    Tensor<T>* b_tensor;
    Tensor<T>* c_tensor;
    //  struct wrapper used for calling spmm routines
    void* b_wrapper;  //  wrapper for b
    void* c_wrapper;  //  wrapper for c
    
    spMatrix_format mat_format;
    #if defined(_HAS_CUDA_)
    math::spgemm_cusparse_settings cusparse_settings;
    #endif

    Tensor<T> *_eval(bool recompute);
    Tensor<T> *_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad);

    private:
    void init_desc(void);
    #if defined(_HAS_CUDA_)
    void init_cusparse_desc(void);
    #endif
};

template <typename T>
spgemmOp<T> *spgemm(??????)


}  //  namespace op
}  //  namespace magmadnn

