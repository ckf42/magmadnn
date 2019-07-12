#pragma once

#include <cassert>
#include <memory>
#include <vector>
#include "sparseMatrix/sparseMatrix.h"
#include "tensor/tensor.h"
#include "utilities_internal.h"

#if defined(_HAS_CUDA_)
#include "cusparse.h"
#endif

namespace magmadnn {

template <typename T>
class graph {
   private:
    sparseMatrix *adjMatrix;
    unsigned V;
    unsigned E;

   public:
    graph(const Tensor<T> *adjMatrixTensorPtr, sparseMat_format format, memory_t mem_type = HOST);
    ~graph(void);
    inline unsigned get_order(void) { return V; }
    inline unsigned get_size(void) { return E; }
    Tensor<T> *get_laplancian(void) const;
    Tensor<T> *get_norm_laplancian(void) const;
};

}  // namespace magmadnn